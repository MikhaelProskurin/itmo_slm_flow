import operator
from typing import Protocol, Callable

from pydantic import BaseModel

from core.router import TFeatureVector

OPERATORS: dict[str, Callable] = {
    "gt":  operator.gt,
    "ge": operator.ge,
    "lt":  operator.lt,
    "le": operator.le,
    "eq": operator.eq,
}

class WeightedRule(BaseModel):
    operator: ...
    threshold: ...
    weight: ...


class Routable(Protocol):

    def call_large_model() -> bool:
        ...


class SLMRoutingPolicy:

    def __init__(self) -> None:
        ...

    def call_large_model(self) -> bool:
        ...


class WeightedRuleBasedRoutingPolicy:

    def __init__(
        self,
        *rules: dict[str, WeightedRule],
        min_triggers: int = 1,
        cumulative_weights_threshold: float = 1.0,
    ) -> None:
        self.rules = rules
        self.min_triggers = min_triggers
        self.cumulative_weights_threshold = cumulative_weights_threshold

    def call_large_model(self, feature_vector: TFeatureVector) -> bool:

        feature_values = feature_vector.model_dump()
        triggered_weight = 0.0
        triggered_count = 0

        for feature_name, (op, threshold, weight) in self.rules.items():
            if feature_name not in feature_values:
                continue
            if OPERATORS[op](feature_values[feature_name], threshold):
                triggered_weight += weight
                triggered_count += 1

        return (
            triggered_count >= self.min_triggers
            and triggered_weight >= self.cumulative_weights_threshold
        )
