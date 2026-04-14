"""
app/api/evaluation.py
----------------------
Endpoints for RAGAS-based pipeline evaluation.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_evaluation_service
from app.core.logger import logger
from app.models.schemas import EvalReport, EvalRequest
from app.services.evaluation import EvaluationService

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/run", response_model=EvalReport)
async def run_evaluation(
    body: EvalRequest,
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
    try:
        return service.run(body.samples)
    except Exception as exc:
        logger.exception("Evaluation run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
