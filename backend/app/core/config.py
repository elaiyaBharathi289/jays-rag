from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
PROJECT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "DocuRAG"
    environment: str = "development"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000

    groq_api_key: str
    groq_model: str = "openai/gpt-oss-120b"

    google_api_key: str
    gemini_embedding_model: str = "gemini-embedding-2"

    database_url: str = "sqlite:///./rag.db"

    chroma_path: str = str(
        PROJECT_DIR / "storage" / "chroma"
    )

    upload_dir: str = str(
        PROJECT_DIR / "storage" / "uploads"
    )

    max_upload_mb: int = 25
    max_files_per_request: int = 5

    chunk_min_chars: int = 350
    chunk_max_chars: int = 1400
    semantic_breakpoint_percentile: int = 85

    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 8
    final_context_k: int = 5
    max_context_chars: int = 18000

    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.92
    semantic_cache_top_k: int = 1
    semantic_cache_ttl_seconds: int = 86400

    # Embedding provider
    embedding_provider: str = "local"

    # Smaller embedding model for low-memory deployment
    local_embedding_model: str = (
        "sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()