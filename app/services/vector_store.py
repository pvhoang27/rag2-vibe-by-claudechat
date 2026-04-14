"""
app/services/vector_store.py
-----------------------------
ChromaDB vector store management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import logger


class VectorStoreService:
    """Manages the ChromaDB vector store lifecycle."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._embeddings = OllamaEmbeddings(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_embed_model,
        )
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self._settings.chroma_collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._settings.chroma_persist_dir,
            )
        return self._store

    def add_documents(self, documents: list[Document]) -> int:
        """Add documents to the vector store. Returns count added."""
        store = self._get_store()
        store.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks to collection '{self._settings.chroma_collection_name}'")
        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """Return (document, score) pairs sorted by relevance."""
        store = self._get_store()
        threshold = score_threshold or self._settings.similarity_threshold
        results = store.similarity_search_with_relevance_scores(query, k=k)
        filtered = [(doc, score) for doc, score in results if score >= threshold]
        logger.debug(f"Retrieved {len(filtered)}/{len(results)} chunks above threshold {threshold}")
        return filtered

    def get_collection_info(self) -> dict:
        store = self._get_store()
        count = store._collection.count()
        return {
            "name": self._settings.chroma_collection_name,
            "document_count": count,
            "embedding_model": self._settings.ollama_embed_model,
        }

    def reset_collection(self) -> None:
        """Delete all documents in the collection."""
        store = self._get_store()
        store._collection.delete(where={"source": {"$ne": ""}})
        logger.warning("Collection reset — all documents deleted")
