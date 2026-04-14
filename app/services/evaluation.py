"""
app/services/evaluation.py
---------------------------
Professional RAG evaluation using RAGAS framework.

Metrics evaluated:
  - faithfulness      : Is the answer grounded in retrieved context?
  - answer_relevancy  : How relevant is the answer to the question?
  - context_recall    : How much of the ground truth is covered by context?
  - context_precision : How precise is the retrieval (no noisy chunks)?
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings

from app.core.config import get_settings
from app.core.logger import logger
from app.models.schemas import EvalProgress, EvalReport, EvalSample, MetricScore
from app.services.rag_chain import RAGChainService


METRIC_DESCRIPTIONS = {
    "faithfulness": "Câu trả lời có căn cứ trong context đã retrieve không (0–1, càng cao càng tốt)",
    "answer_relevancy": "Câu trả lời có liên quan đến câu hỏi không (0–1, càng cao càng tốt)",
    "context_recall": "Context retrieve được bao phủ ground truth đến đâu (0–1, càng cao càng tốt)",
    "context_precision": "Context retrieve có chính xác, không có nhiễu không (0–1, càng cao càng tốt)",
}

METRIC_ALIASES = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "answer_relevance", "response_relevancy", "response_relevance"],
    "context_recall": ["context_recall", "context_entity_recall"],
    "context_precision": [
        "context_precision",
        "context_precision_without_reference",
        "context_precision_with_reference",
    ],
}

FAST_METRICS = [answer_relevancy]
FULL_METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]


class EvaluationService:
    """Orchestrates RAGAS-based evaluation of the RAG pipeline."""

    def __init__(self, rag_chain: RAGChainService) -> None:
        self._settings = get_settings()
        self._rag_chain = rag_chain
        self._progress_lock = threading.Lock()
        self._progress = EvalProgress(
            is_running=False,
            stage="idle",
            percent=0.0,
            completed_samples=0,
            total_samples=0,
            message="Ready",
        )
        # Wrap Ollama models for RAGAS
        self._llm_wrapper = LangchainLLMWrapper(
            OllamaLLM(
                base_url=self._settings.ollama_base_url,
                model=self._settings.ollama_llm_model,
                temperature=0,
            )
        )
        self._embed_wrapper = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(
                base_url=self._settings.ollama_base_url,
                model=self._settings.ollama_embed_model,
            )
        )

    def run(self, samples: list[EvalSample], mode: str = "full", output_tag: str | None = None) -> EvalReport:
        """
        Execute full RAGAS evaluation pipeline.

        Steps:
          1. For each sample, run RAG chain to get answer + context.
          2. Build HuggingFace Dataset expected by RAGAS.
          3. Run RAGAS metrics with local Ollama models.
          4. Aggregate scores into EvalReport and persist to disk.
        """
        run_mode = (mode or "full").strip().lower()
        if run_mode not in {"full", "fast"}:
            raise ValueError("mode must be 'full' or 'fast'")

        logger.info(f"Starting evaluation on {len(samples)} samples (mode={run_mode}) …")
        self._set_progress(
            is_running=True,
            stage="retrieving",
            percent=0.0,
            completed_samples=0,
            total_samples=len(samples),
            message=f"Preparing evaluation ({run_mode} mode)",
        )

        try:
            rows = self._build_eval_rows(samples)
            dataset = Dataset.from_list(rows)

            self._set_progress(
                stage="scoring",
                percent=75.0,
                message="Running RAGAS metrics",
            )

            metrics = FAST_METRICS if run_mode == "fast" else FULL_METRICS
            for metric in metrics:
                metric.llm = self._llm_wrapper
                if hasattr(metric, "embeddings"):
                    metric.embeddings = self._embed_wrapper

            logger.info("Running RAGAS evaluation — this may take a few minutes …")
            result = evaluate(dataset=dataset, metrics=metrics)

            self._set_progress(
                stage="finalizing",
                percent=95.0,
                message="Persisting report",
            )

            report = self._build_report(result, sample_count=len(samples), selected_metrics=metrics)
            self._persist_report(report, rows, mode=run_mode, output_tag=output_tag)
            logger.info(
                f"Evaluation complete | overall_score={report.overall_score:.4f} | "
                f"saved to {report.output_path}"
            )
            self._set_progress(
                is_running=False,
                stage="done",
                percent=100.0,
                completed_samples=len(samples),
                total_samples=len(samples),
                message="Evaluation complete",
            )
            return report
        except Exception as exc:
            self._set_progress(
                is_running=False,
                stage="failed",
                message=f"Evaluation failed: {exc}",
            )
            raise

    def get_progress(self) -> EvalProgress:
        with self._progress_lock:
            return self._progress.model_copy(deep=True)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_eval_rows(self, samples: list[EvalSample]) -> list[dict]:
        rows = []
        total = max(len(samples), 1)
        for i, sample in enumerate(samples):
            self._set_progress(
                stage="retrieving",
                percent=round((((i + 0.5) / total) * 70.0), 2),
                completed_samples=i,
                total_samples=len(samples),
                message=f"Retrieving context for sample {i + 1}/{len(samples)}",
            )
            logger.debug(f"  [{i + 1}/{len(samples)}] Running RAG for: {sample.question[:60]}…")
            response = self._rag_chain.query(sample.question)
            context = [s.content for s in response.sources] if response.sources else [""]
            rows.append(
                {
                    "question": sample.question,
                    "answer": response.answer,
                    "contexts": context,
                    "ground_truth": sample.ground_truth,
                }
            )
            self._set_progress(
                stage="retrieving",
                percent=round((((i + 1) / total) * 70.0), 2),
                completed_samples=i + 1,
                total_samples=len(samples),
                message=f"Retrieved sample {i + 1}/{len(samples)}",
            )
        return rows

    def _set_progress(
        self,
        *,
        is_running: bool | None = None,
        stage: str | None = None,
        percent: float | None = None,
        completed_samples: int | None = None,
        total_samples: int | None = None,
        message: str | None = None,
    ) -> None:
        with self._progress_lock:
            current = self._progress.model_dump()
            if is_running is not None:
                current["is_running"] = is_running
            if stage is not None:
                current["stage"] = stage
            if percent is not None:
                current["percent"] = max(0.0, min(100.0, float(percent)))
            if completed_samples is not None:
                current["completed_samples"] = max(0, completed_samples)
            if total_samples is not None:
                current["total_samples"] = max(0, total_samples)
            if message is not None:
                current["message"] = message
            current["updated_at"] = datetime.utcnow()
            self._progress = EvalProgress(**current)

    def _build_report(self, ragas_result, sample_count: int, selected_metrics: list) -> EvalReport:
        metric_scores: list[MetricScore] = []
        raw_scores: list[float] = []

        scores_by_metric: dict[str, float] = {}
        selected_metric_names = {getattr(m, "name", "") for m in selected_metrics}

        # RAGAS return type differs across versions (dict vs EvaluationResult).
        if isinstance(ragas_result, dict):
            for k, v in ragas_result.items():
                try:
                    scores_by_metric[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
        else:
            as_dict = None
            if hasattr(ragas_result, "to_dict"):
                try:
                    as_dict = ragas_result.to_dict()
                except Exception:
                    as_dict = None

            if isinstance(as_dict, dict):
                for k, v in as_dict.items():
                    try:
                        scores_by_metric[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue

            # Fallback: compute mean scores from per-sample dataframe.
            if not scores_by_metric and hasattr(ragas_result, "to_pandas"):
                try:
                    df = ragas_result.to_pandas()
                    for col in df.columns:
                        value = pd.to_numeric(df[col], errors="coerce").mean()
                        if pd.notna(value):
                            scores_by_metric[str(col)] = float(value)
                except Exception:
                    pass

        normalized = self._normalize_metric_scores(scores_by_metric)
        if not normalized:
            logger.warning(
                "RAGAS returned no usable numeric metrics. Falling back to 0.0 scores. Raw keys=%s",
                sorted(scores_by_metric.keys()),
            )

        for metric_name, description in METRIC_DESCRIPTIONS.items():
            if selected_metric_names and metric_name not in selected_metric_names:
                continue
            score = float(normalized.get(metric_name, 0.0) or 0.0)
            raw_scores.append(score)
            metric_scores.append(
                MetricScore(
                    name=metric_name,
                    score=round(score, 4),
                    description=description,
                )
            )

        overall = round(sum(raw_scores) / len(raw_scores), 4) if raw_scores else 0.0

        return EvalReport(
            metrics=metric_scores,
            overall_score=overall,
            sample_count=sample_count,
            model=self._settings.ollama_llm_model,
        )

    def _normalize_metric_scores(self, raw: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        lowered = {str(k).strip().lower(): v for k, v in raw.items()}

        for canonical, aliases in METRIC_ALIASES.items():
            for alias in aliases:
                if alias in lowered:
                    try:
                        value = float(lowered[alias])
                    except (TypeError, ValueError):
                        continue
                    if not pd.notna(value):
                        logger.warning("Metric '%s' returned NaN; treating as 0.0", canonical)
                        value = 0.0
                    # Keep scores in expected range to avoid noisy displays from upstream libs.
                    value = max(0.0, min(1.0, value))
                    normalized[canonical] = value
                    break
        return normalized

    def _persist_report(self, report: EvalReport, rows: list[dict], mode: str, output_tag: str | None) -> None:
        out_dir = Path(self._settings.eval_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_mode = re.sub(r"[^a-zA-Z0-9_-]", "", mode) or "full"
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]", "", (output_tag or "").strip())
        suffix = f"_{safe_mode}"
        if safe_tag:
            suffix += f"_{safe_tag}"
        json_path = out_dir / f"eval_report_{timestamp}{suffix}.json"
        csv_path = out_dir / f"eval_detail_{timestamp}{suffix}.csv"

        report.output_path = str(json_path)

        # Save JSON report
        json_path.write_text(
            json.dumps(report.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

        # Save per-sample CSV
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
