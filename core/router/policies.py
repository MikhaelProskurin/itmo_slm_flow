"""Routing policies that decide whether a request should go to the SLM or LLM.

Each policy implements ``BaseRoutingPolicy`` and returns ``True`` to escalate to
the LLM, or ``False`` to keep it with the SLM.
"""

import operator
from typing import Protocol, Any

from core.router import TFeatureVector

_OPERATORS: dict[str, Any] = {
    "gt":  operator.gt,
    "ge": operator.ge,
    "lt":  operator.lt,
    "le": operator.le,
    "eq": operator.eq,
}


class Routable(Protocol):

    def call_large_model(feature_vector: TFeatureVector) -> bool:
        ...


class SLMRoutingPolicy:

    def __init__(self) -> None:
        ...
    
    def call_large_model(self) -> bool:
        ...


class ThresholdRoutingPolicy:
    """Routes to LLM when weighted sum of triggered rules exceeds total_threshold.

    Args:
        rules: Mapping of feature name to ``(operator, threshold, weight)`` tuples.
               Supported operators: ``"gt"``, ``"ge"``, ``"lt"``, ``"le"``, ``"eq"``.
        total_threshold: Minimum cumulative weight to route to LLM.
        min_triggers: Minimum number of rules that must fire to route to LLM.

    Example::

        policy = ThresholdRoutingPolicy(
            rules={
                "query_token_count":   ("gt", 50.0,  0.3),
                "avg_lexical_overlap": ("lt", 0.15,  0.6),
                "inter_doc_similarity":("gt", 0.85,  0.5),
            },
            total_threshold=1.0,
            min_triggers=2,
        )
    """

    def __init__(
        self,
        rules: dict[str, tuple[str, float, float]],
        total_threshold: float = 1.0,
        min_triggers: int = 2,
    ) -> None:
        self.rules = rules
        self.total_threshold = total_threshold
        self.min_triggers = min_triggers

    def decide(self, features: FeatureVector) -> bool:
        """Return True if weighted triggers exceed total_threshold AND min_triggers is met."""
        feature_values = features.model_dump()
        triggered_weight = 0.0
        triggered_count = 0

        for feature_name, (operator, threshold, weight) in self.rules.items():
            if feature_name not in feature_values:
                continue

            if _OPERATORS[operator](feature_values[feature_name], threshold):
                triggered_weight += weight
                triggered_count += 1

        return triggered_count >= self.min_triggers and triggered_weight >= self.total_threshold
