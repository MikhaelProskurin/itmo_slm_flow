"""Post-evaluation metrics for assessing SLM routing efficiency across an evaluation run."""

from typing import Literal
from pydantic import BaseModel

from core.pipeline.runner import (
    EvaluationRecord,
    RerankingMetrics,
    CompressionMetrics
)


class SLMRoutingMetrics(BaseModel):
    """Aggregate routing efficiency metrics for the small model across an evaluation run."""

    slm_success_ratio: float
    slm_routing_ratio: float


class EvaluationSummary(BaseModel):
    """Mean answer-quality metrics aggregated by task type across an evaluation run."""

    reranking_avg_bert_f1: float
    reranking_avg_exact_match: float
    reranking_avg_jscore: float
    compression_avg_bert_f1: float
    compression_avg_rouge_l: float
    compression_avg_rouge_n: float
    compression_avg_compression_ratio: float
    compression_avg_jscore: float


def compute_slm_routing_metrics(records: list[EvaluationRecord], threshold: float = 4.0) -> SLMRoutingMetrics:
    """Compute SLM routing efficiency from a list of evaluated records.

    Args:
        records: Fully evaluated inference records produced by ``RAGPipelineRunner.aevaluate``.
        threshold: Minimum ``jscore.final_score`` to count an SLM call as successful.

    Returns:
        ``SLMRoutingMetrics`` with ``slm_success_ratio`` (successful SLM calls / total SLM calls)
        and ``slm_routing_ratio`` (SLM calls / all calls).
    """
    _routing: Literal["_slm"] = "_slm"

    small_model_calls = list(filter(lambda row: row.routing == _routing, records))
    success_small_model_calls = list(filter(lambda row: row.jscore.final_score >= threshold, small_model_calls))

    return SLMRoutingMetrics(
        slm_success_ratio=len(success_small_model_calls) / len(small_model_calls),
        slm_routing_ratio=len(small_model_calls) / len(records)
    )


def get_evaluation_summary(records: list[EvaluationRecord]) -> EvaluationSummary:
    """Aggregate answer-quality metrics across all evaluated records by task type.

    Args:
        records: Fully evaluated inference records produced by ``RAGPipelineRunner.aevaluate``.

    Returns:
        ``EvaluationSummary`` with per-task mean metrics across all matching records.
    """
    _reranking = [r for r in records if isinstance(r.answer_metrics, RerankingMetrics)]
    _compression = [r for r in records if isinstance(r.answer_metrics, CompressionMetrics)]

    average_fn = lambda values: sum(values) / len(values) if values else 0.0

    return EvaluationSummary(
        reranking_avg_bert_f1=average_fn([r.answer_metrics.bert_f1 for r in _reranking]),
        reranking_avg_exact_match=average_fn([1.0 if r.answer_metrics.exact_match else 0.0 for r in _reranking]),
        reranking_avg_jscore=average_fn([r.jscore.final_score for r in _reranking]),
        compression_avg_bert_f1=average_fn([r.answer_metrics.bert_f1 for r in _compression]),
        compression_avg_rouge_l=average_fn([r.answer_metrics.rouge_l for r in _compression]),
        compression_avg_rouge_n=average_fn([r.answer_metrics.rouge_n for r in _compression]),
        compression_avg_compression_ratio=average_fn([r.answer_metrics.compression_ratio for r in _compression]),
        compression_avg_jscore=average_fn([r.jscore.final_score for r in _compression])
    )
