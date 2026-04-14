"""
app/api/ingest.py
-----------------
Endpoints for document ingestion.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.core.dependencies import get_ingestion_service
from app.core.logger import logger
from app.models.schemas import IngestResponse
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(
    file: UploadFile = File(...),
    service: Annotated[IngestionService, Depends(get_ingestion_service)] = None,
) -> IngestResponse:
    """Upload and ingest a single document (PDF, TXT, DOCX, MD)."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    settings = get_settings()
    dest = Path(settings.data_raw_dir) / file.filename
    try:
        with dest.open("wb") as fp:
            shutil.copyfileobj(file.file, fp)
        count = service.ingest_file(dest)
    except Exception as exc:
        logger.exception(f"Ingestion failed for {file.filename}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message=f"File '{file.filename}' ingested successfully.",
        documents_added=count,
        collection=settings.chroma_collection_name,
    )


@router.post("/directory", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_directory(
    service: Annotated[IngestionService, Depends(get_ingestion_service)] = None,
) -> IngestResponse:
    """Ingest all supported documents from the configured raw data directory."""
    settings = get_settings()
    try:
        count = service.ingest_directory()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Directory ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        message="Directory ingested successfully.",
        documents_added=count,
        collection=settings.chroma_collection_name,
    )
