"""
app/core/dependencies.py
------------------------
FastAPI dependency injection — singleton service instances.
"""

from functools import lru_cache

from app.services.vector_store import VectorStoreService
from app.services.ingestion import IngestionService
from app.services.rag_chain import RAGChainService
from app.services.evaluation import EvaluationService


@lru_cache
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(vector_store=get_vector_store())


@lru_cache
def get_rag_chain() -> RAGChainService:
    return RAGChainService(vector_store=get_vector_store())


@lru_cache
def get_evaluation_service() -> EvaluationService:
    return EvaluationService(rag_chain=get_rag_chain())
