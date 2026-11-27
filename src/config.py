from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL 17
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "langveyor"
    postgres_password: str = "langveyor"
    postgres_db: str = "langveyor"
    database_url: str = "postgresql+asyncpg://langveyor:langveyor@localhost:5432/langveyor"
    checkpointer_dsn: str = "postgresql://langveyor:langveyor@localhost:5432/langveyor"

    # Qdrant & Local CPU Embeddings
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mes_sops"
    embedding_model: str = "cointegrated/LaBSE-en-ru"
    embedding_device: str = "cpu"

    # OpenRouter LLM (4 distinct roles; Auditor and Judge use separate model families from Investigator)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model_fast: str = "google/gemini-2.5-flash"
    openrouter_model_investigator: str = "deepseek/deepseek-v3.2"
    openrouter_model_auditor: str = "openai/gpt-4.1-mini"
    openrouter_model_judge: str = "anthropic/claude-haiku-4.5"

    # Agent & RAG parameters
    max_tool_steps: int = 8
    max_revisions: int = 2
    rag_top_k: int = 5
    rag_dense_weight: float = 0.55
    rag_bm25_weight: float = 0.45
    sampling_hmac_secret: str = "langveyor-alg001-deterministic-seed-2026"

    # Knowledge Base paths
    kb_dir: Path = PROJECT_ROOT / "knowledge_base"
    bpmn_path: Path = PROJECT_ROOT / "knowledge_base" / "support_process.bpmn"
    sops_dir: Path = PROJECT_ROOT / "knowledge_base" / "sops"

    @property
    def has_openrouter_key(self) -> bool:
        key = self.openrouter_api_key.strip()
        return bool(key) and not key.startswith("sk-or-v1-placeholder")


def get_settings() -> Settings:
    """Return a freshly loaded Settings instance."""
    return Settings()
