"""
app/api/evaluation.py
----------------------
Endpoints for RAGAS-based pipeline evaluation.
"""

from __future__ import annotations

from functools import partial
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.core.dependencies import get_evaluation_service
from app.core.logger import logger
from app.models.schemas import EvalProgress, EvalReport, EvalRequest
from app.services.evaluation import EvaluationService

router = APIRouter(prefix="/eval", tags=["Evaluation"])


def _is_ollama_connection_error(message: str) -> bool:
    msg = message.lower()
    indicators = [
        "localhost",
        "11434",
        "connection refused",
        "winerror 10061",
        "/api/embeddings",
        "inference endpoint",
    ]
    return all(token in msg for token in ["11434", "connection"]) or any(
        token in msg for token in indicators
    )


@router.post("/run", response_model=EvalReport)
async def run_evaluation(
    body: EvalRequest,
    mode: str = "full",
    tag: str | None = None,
    service: Annotated[EvaluationService, Depends(get_evaluation_service)] = None,
) -> EvalReport:
    """
    Run RAGAS evaluation on a provided list of (question, ground_truth) pairs.
    
    The pipeline will:
      1. Run each question through the RAG chain.
      2. Evaluate faithfulness, answer_relevancy, context_recall, context_precision.
      3. Return a structured report and persist results to disk.
    """
    if not body.samples:
        raise HTTPException(status_code=422, detail="samples list cannot be empty")
    run_mode = (mode or "full").strip().lower()
    if run_mode not in {"full", "fast"}:
        raise HTTPException(status_code=422, detail="mode must be 'full' or 'fast'")
    try:
        return await run_in_threadpool(partial(service.run, body.samples, mode=run_mode, output_tag=tag))
    except Exception as exc:
        logger.exception("Evaluation run failed")
        if _is_ollama_connection_error(str(exc)):
            settings = get_settings()
            detail = (
                "Ollama is not reachable for embeddings. "
                f"Expected endpoint: {settings.ollama_base_url}. "
                f"Please start Ollama and ensure model '{settings.ollama_embed_model}' is available. "
                "Try: 'ollama serve' (if not running), then 'ollama pull nomic-embed-text'."
            )
            raise HTTPException(status_code=503, detail=detail) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/progress", response_model=EvalProgress)
async def evaluation_progress(
    service: Annotated[EvaluationService, Depends(get_evaluation_service)] = None,
) -> EvalProgress:
    """Return current evaluation progress for CLI polling."""
    return service.get_progress()
