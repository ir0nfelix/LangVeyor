from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from uuid import UUID, uuid5

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import Settings, get_settings

SOP_NAMESPACE = UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


@dataclass(frozen=True, slots=True)
class SopChunk:
    chunk_id: str
    point_id: str
    source_file: str
    doc_title: str
    section_title: str
    content: str


def chunk_markdown_sop(file_path: Path) -> list[SopChunk]:
    """Split a Markdown SOP file into semantic section chunks preserving doc & section titles."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc_title = file_path.stem
    for line in lines:
        if line.startswith("# "):
            doc_title = line[2:].strip()
            break

    sections: list[tuple[str, list[str]]] = []
    cur_header = doc_title
    cur_buf: list[str] = []

    for line in lines:
        if re.match(r"^#{2,3}\s+", line):
            body = "\n".join(cur_buf).strip()
            if body and not body.startswith("# "):
                sections.append((cur_header, cur_buf))
            cur_header = re.sub(r"^#{2,3}\s+", "", line).strip()
            cur_buf = [line]
        else:
            if line.startswith("# "):
                continue
            cur_buf.append(line)

    if cur_buf:
        body = "\n".join(cur_buf).strip()
        if body:
            sections.append((cur_header, cur_buf))

    chunks: list[SopChunk] = []
    for idx, (sec_title, buf) in enumerate(sections, start=1):
        sec_text = "\n".join(buf).strip()
        full_content = f"{doc_title} | {sec_title}\n{sec_text}"
        chunk_key = f"{file_path.name}::{idx}::{sec_title}"
        point_uuid = str(uuid5(SOP_NAMESPACE, chunk_key))
        chunks.append(
            SopChunk(
                chunk_id=chunk_key,
                point_id=point_uuid,
                source_file=file_path.name,
                doc_title=doc_title,
                section_title=sec_title,
                content=full_content,
            )
        )
    return chunks


def load_all_sop_chunks(sops_dir: Path) -> list[SopChunk]:
    if not sops_dir.exists():
        raise FileNotFoundError(f"SOPs directory not found: {sops_dir}")
    chunks: list[SopChunk] = []
    for md_path in sorted(sops_dir.glob("*.md")):
        chunks.extend(chunk_markdown_sop(md_path))
    return chunks


@lru_cache(maxsize=2)
def get_embedder(model_name: str, device: str = "cpu") -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


def index_sops(
    settings: Settings | None = None, *, recreate: bool = True
) -> dict[str, int | float | str]:
    """Chunk SOP markdown files, compute local CPU LaBSE embeddings, and upsert to Qdrant."""
    t0 = time.perf_counter()
    cfg = settings or get_settings()
    chunks = load_all_sop_chunks(cfg.sops_dir)
    if not chunks:
        raise RuntimeError(f"No SOP markdown files found in {cfg.sops_dir}")

    embedder = get_embedder(cfg.embedding_model, cfg.embedding_device)
    vectors = embedder.encode(
        [c.content for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dim = int(vectors.shape[1])

    client = QdrantClient(url=cfg.qdrant_url)
    exists = client.collection_exists(cfg.qdrant_collection)
    if recreate and exists:
        client.delete_collection(cfg.qdrant_collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    points = [
        models.PointStruct(
            id=chunk.point_id,
            vector=vec.tolist(),
            payload={
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "doc_title": chunk.doc_title,
                "section_title": chunk.section_title,
                "content": chunk.content,
            },
        )
        for chunk, vec in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=cfg.qdrant_collection, points=points, wait=True)

    elapsed = time.perf_counter() - t0
    return {
        "collection": cfg.qdrant_collection,
        "chunks_indexed": len(points),
        "vector_dim": dim,
        "elapsed_sec": round(elapsed, 2),
    }


if __name__ == "__main__":
    from src.rag.hybrid_retriever import HybridRetriever

    cfg = get_settings()
    stats = index_sops(cfg, recreate=True)
    print(f"Indexed Qdrant collection '{stats['collection']}': {stats}")

    retriever = HybridRetriever(cfg)
    sample_hits = retriever.search("ёмкость АКБ 6S ниже допуска 33 Ач и отзыв партии", top_k=3)
    print(f"Verification hybrid query returned {len(sample_hits)} chunks:")
    for hit in sample_hits:
        print(
            f"  - [{hit.source_file}] {hit.section_title} "
            f"(rrf={hit.rrf_score:.4f}, dense={hit.dense_score:.4f}, bm25={hit.bm25_score:.2f})"
        )
