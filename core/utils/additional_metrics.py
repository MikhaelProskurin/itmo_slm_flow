"""Post-evaluation metrics for assessing SLM routing efficiency across an evaluation run."""

from pydantic import BaseModel

from core.pipeline.runner import EvaluationRecord
from core.router import TRoute


class SLMRoutingMetrics(BaseModel):
    """Aggregate routing efficiency metrics for the small model across an evaluation run."""

    slm_success_ratio: float
    slm_routing_ratio: float


def compute_slm_routing_metrics(records: list[EvaluationRecord], threshold: float = 4.0) -> SLMRoutingMetrics:
    """Compute SLM routing efficiency from a list of evaluated records.

    Args:
        records: Fully evaluated inference records produced by ``RAGPipelineRunner.aevaluate``.
        threshold: Minimum ``jscore.final_score`` to count an SLM call as successful.

    Returns:
        ``SLMRoutingMetrics`` with ``slm_success_ratio`` (successful SLM calls / total SLM calls)
        and ``slm_routing_ratio`` (SLM calls / all calls).
    """
    _routing: TRoute = "_slm"

    small_model_calls = list(filter(lambda row: row.routing == _routing, records))
    success_small_model_calls = list(filter(lambda row: row.jscore.final_score >= threshold, records))

    return SLMRoutingMetrics(
        slm_success_ratio=len(success_small_model_calls) / len(small_model_calls),
        slm_routing_ratio=len(small_model_calls) / len(records)
    )
