"""
app/api/chat.py
---------------
Chat and collection management endpoints.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_rag_chain, get_vector_store
from app.core.logger import logger
from app.models.schemas import ChatRequest, ChatResponse, CollectionInfo
from app.services.rag_chain import RAGChainService
from app.services.vector_store import VectorStoreService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/query", response_model=ChatResponse)
async def query(
    body: ChatRequest,
    rag: Annotated[RAGChainService, Depends(get_rag_chain)] = None,
) -> ChatResponse:
    """Submit a question and receive a RAG-generated answer with sources."""
    try:
        return rag.query(question=body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception(f"Query failed: {body.question[:80]}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/collection", response_model=CollectionInfo)
async def collection_info(
    vs: Annotated[VectorStoreService, Depends(get_vector_store)] = None,
) -> CollectionInfo:
    """Return metadata about the current vector store collection."""
    info = vs.get_collection_info()
    return CollectionInfo(**info)


@router.delete("/collection", status_code=204)
async def reset_collection(
    vs: Annotated[VectorStoreService, Depends(get_vector_store)] = None,
) -> None:
    """Delete all documents from the vector store (destructive!)."""
    try:
        vs.reset_collection()
    except Exception as exc:
        logger.exception("Collection reset failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
