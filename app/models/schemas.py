"""
app/models/schemas.py
---------------------
Pydantic v2 request / response schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ─── Ingest ────────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    documents_added: int
    collection: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Chat ──────────────────────────────────────────────────────────────────────

class Source(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: Optional[float] = None


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[dict] = Field(default_factory=list)
    top_k: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Collection ────────────────────────────────────────────────────────────────

class CollectionInfo(BaseModel):
    name: str
    document_count: int
    embedding_model: str


# ─── Evaluation ────────────────────────────────────────────────────────────────

class EvalSample(BaseModel):
    question: str
    ground_truth: str
    context: Optional[list[str]] = None


class EvalRequest(BaseModel):
    samples: list[EvalSample]


class MetricScore(BaseModel):
    name: str
    score: float
    description: str


class EvalReport(BaseModel):
    metrics: list[MetricScore]
    overall_score: float
    sample_count: int
    model: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    output_path: Optional[str] = None


class EvalProgress(BaseModel):
    is_running: bool
    stage: str
    percent: float = Field(ge=0.0, le=100.0)
    completed_samples: int = 0
    total_samples: int = 0
    message: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
