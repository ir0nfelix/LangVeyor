from __future__ import annotations

import re
from dataclasses import dataclass

from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from src.config import Settings, get_settings
from src.rag.indexer import SopChunk, get_embedder, load_all_sop_chunks

TOKEN_RE = re.compile(r"[a-zа-яё0-9]+(?:[-_][a-zа-яё0-9]+)*", re.IGNORECASE)


def tokenize_bilingual(text: str) -> list[str]:
    """Tokenize Russian/English text while keeping both full part codes and split sub-tokens."""
    tokens: list[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if "-" in match or "_" in match:
            for part in re.split(r"[-_]+", match):
                if part and part != match:
                    tokens.append(part)
    return tokens


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    source_file: str
    doc_title: str
    section_title: str
    content: str
    rrf_score: float
    dense_score: float
    bm25_score: float


class HybridRetriever:
    """Hybrid retriever combining Qdrant (cointegrated/LaBSE-en-ru) + BM25 via RRF."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chunks: list[SopChunk] | None = None,
        qdrant_client: QdrantClient | None = None,
        use_dense: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self._chunks: list[SopChunk] = (
            chunks if chunks is not None else load_all_sop_chunks(self.settings.sops_dir)
        )
        self._chunk_by_id: dict[str, SopChunk] = {c.chunk_id: c for c in self._chunks}
        self._tokenized_corpus = [tokenize_bilingual(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        self._use_dense = use_dense
        self._qdrant = (
            qdrant_client
            if qdrant_client is not None
            else (QdrantClient(url=self.settings.qdrant_url) if use_dense else None)
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        k = top_k or self.settings.rag_top_k
        if not self._chunks:
            return []
        candidate_k = min(len(self._chunks), max(k * 3, 20))

        # 1. BM25 sparse ranking
        q_tokens = tokenize_bilingual(query)
        bm25_scores_arr = self._bm25.get_scores(q_tokens) if q_tokens else [0.0] * len(self._chunks)
        bm25_ranked = sorted(
            enumerate(bm25_scores_arr),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:candidate_k]

        bm25_ranks: dict[str, int] = {}
        bm25_score_map: dict[str, float] = {}
        for rank_idx, (doc_idx, score_val) in enumerate(bm25_ranked, start=1):
            cid = self._chunks[doc_idx].chunk_id
            bm25_ranks[cid] = rank_idx
            bm25_score_map[cid] = float(score_val)

        # 2. Qdrant dense ranking (LaBSE-en-ru CPU embeddings)
        dense_ranks: dict[str, int] = {}
        dense_score_map: dict[str, float] = {}
        if self._use_dense and self._qdrant is not None:
            embedder = get_embedder(self.settings.embedding_model, self.settings.embedding_device)
            q_vec = embedder.encode(
                query, normalize_embeddings=True, show_progress_bar=False
            ).tolist()
            resp = self._qdrant.query_points(
                collection_name=self.settings.qdrant_collection,
                query=q_vec,
                limit=candidate_k,
                with_payload=True,
            )
            for rank_idx, pt in enumerate(resp.points, start=1):
                payload = pt.payload or {}
                cid = str(payload.get("chunk_id") or "")
                if cid:
                    dense_ranks[cid] = rank_idx
                    dense_score_map[cid] = float(pt.score)
                    if cid not in self._chunk_by_id:
                        self._chunk_by_id[cid] = SopChunk(
                            chunk_id=cid,
                            point_id=str(pt.id),
                            source_file=str(payload.get("source_file", "")),
                            doc_title=str(payload.get("doc_title", "")),
                            section_title=str(payload.get("section_title", "")),
                            content=str(payload.get("content", "")),
                        )

        # 3. Reciprocal Rank Fusion (RRF)
        all_ids = set(bm25_ranks.keys()) | set(dense_ranks.keys())
        w_dense = self.settings.rag_dense_weight if dense_ranks else 0.0
        w_bm25 = self.settings.rag_bm25_weight if bm25_ranks else 0.0

        fused: list[RetrievedChunk] = []
        for cid in all_ids:
            chunk = self._chunk_by_id.get(cid)
            if chunk is None:
                continue
            r_dense = dense_ranks.get(cid)
            r_bm25 = bm25_ranks.get(cid)
            rrf = (w_dense / (rrf_k + r_dense) if r_dense is not None else 0.0) + (
                w_bm25 / (rrf_k + r_bm25) if r_bm25 is not None else 0.0
            )
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    source_file=chunk.source_file,
                    doc_title=chunk.doc_title,
                    section_title=chunk.section_title,
                    content=chunk.content,
                    rrf_score=round(rrf, 6),
                    dense_score=round(dense_score_map.get(cid, 0.0), 6),
                    bm25_score=round(bm25_score_map.get(cid, 0.0), 6),
                )
            )

        fused.sort(key=lambda c: c.rrf_score, reverse=True)
        return fused[:k]
