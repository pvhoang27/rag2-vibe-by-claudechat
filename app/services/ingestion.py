"""
app/services/ingestion.py
--------------------------
Document loading, splitting, and ingestion pipeline.
Supports: PDF, TXT, DOCX, MD
"""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import logger
from app.services.vector_store import VectorStoreService


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".md": UnstructuredMarkdownLoader,
}


class IngestionService:
    """Orchestrates end-to-end document ingestion."""

    def __init__(self, vector_store: VectorStoreService) -> None:
        self._settings = get_settings()
        self._vector_store = vector_store
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_file(self, file_path: str | Path) -> int:
        """Load a single file, split it, and store in vector DB."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        docs = self._load_file(path)
        logger.info(f"Loaded {len(docs)} docs from {path.name}")
        chunks = self._split(docs, source=str(path))
        logger.info(f"Split into {len(chunks)} chunks from {path.name}")
        if not chunks:
            logger.warning(f"No chunks created from {path.name}")
        count = self._vector_store.add_documents(chunks)
        logger.info(f"Added {count} chunks to vector store from {path.name}")
        return count

    def ingest_directory(self, dir_path: str | Path | None = None) -> int:
        """Ingest all supported files in a directory."""
        directory = Path(dir_path or self._settings.data_raw_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        total = 0
        files = [
            f for f in directory.rglob("*")
            if f.suffix.lower() in LOADER_MAP and f.is_file()
        ]
        logger.info(f"Found {len(files)} supported file(s) in {directory}")

        for file in files:
            try:
                logger.info(f"Ingesting file: {file}")
                count = self.ingest_file(file)
                total += count
                logger.info(f"  ✓ {file.name} → {count} chunks")
            except Exception as exc:
                logger.error(f"  ✗ {file.name} failed: {exc}")

        logger.info(f"Total chunks added from directory: {total}")
        return total

    def ingest_text(self, text: str, source: str = "manual") -> int:
        """Ingest raw text directly (useful for testing)."""
        doc = Document(page_content=text, metadata={"source": source})
        chunks = self._split([doc], source=source)
        return self._vector_store.add_documents(chunks)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> list[Document]:
        loader_cls = LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        try:
            # Nếu là file .txt, thử đọc bằng Python thuần nếu TextLoader lỗi
            if path.suffix.lower() == ".txt":
                try:
                    loader = loader_cls(str(path))
                    docs = loader.load()
                    logger.debug(f"Loaded {len(docs)} page(s) from {path.name} (TextLoader)")
                    return docs
                except Exception as exc:
                    logger.warning(f"TextLoader failed for {path}, thử đọc bằng Python thuần: {exc}")
                    try:
                        with open(path, encoding="utf-8") as f:
                            text = f.read()
                        from langchain_core.documents import Document
                        doc = Document(page_content=text, metadata={"source": str(path)})
                        logger.info(f"Loaded 1 page from {path.name} (python open)")
                        return [doc]
                    except Exception as exc2:
                        logger.error(f"Cả TextLoader và open() đều lỗi với {path}: {exc2}", exc_info=True)
                        raise
            else:
                loader = loader_cls(str(path))
                docs = loader.load()
                logger.debug(f"Loaded {len(docs)} page(s) from {path.name}")
                return docs
        except Exception as exc:
            logger.error(f"Exception when loading {path}: {exc}", exc_info=True)
            raise

    def _split(self, docs: list[Document], source: str) -> list[Document]:
        chunks = self._splitter.split_documents(docs)
        # Enrich metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("source", source)
            chunk.metadata["chunk_index"] = i
        return chunks
