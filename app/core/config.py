"""
app/core/config.py
------------------
Centralised configuration using Pydantic-Settings.
All values are read from environment variables / .env file.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="RAG Chatbot")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_llm_model: str = Field(default="llama3.2:3b")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    chroma_collection_name: str = Field(default="rag_collection")

    # RAG
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    top_k_retrieval: int = Field(default=3)
    similarity_threshold: float = Field(default=0.3)

    # Data paths
    data_raw_dir: str = Field(default="./data/raw")
    data_processed_dir: str = Field(default="./data/processed")

    # Evaluation
    eval_output_dir: str = Field(default="./data/eval_results")
    eval_sample_size: int = Field(default=10)

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for dir_path in [
            self.chroma_persist_dir,
            self.data_raw_dir,
            self.data_processed_dir,
            self.eval_output_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance (cached)."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
