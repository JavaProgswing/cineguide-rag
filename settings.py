"""Environment-backed configuration for CineGuide."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    database_path: Path = Path(os.getenv("DATABASE_PATH", "data/cineguide.db"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    tmdb_api_key: str | None = os.getenv("TMDB_API_KEY") or None
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or None
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-5.4-mini")

    @property
    def documents_path(self) -> Path:
        return self.data_dir / "documents.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / "embeddings.npy"

    @property
    def index_metadata_path(self) -> Path:
        return self.data_dir / "index_metadata.json"


settings = Settings()
