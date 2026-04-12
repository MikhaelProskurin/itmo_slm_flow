from typing import Literal

from core.pipeline import TRoutingMode
from core.tasks import RAGTask
from core.router import (
    Routable,
    TFeatureVector,
    RAGFeatureExtractor,
)

TRoute = Literal["_slm", "_llm"]

TModelSelection = tuple[TFeatureVector, TRoute]

class LMRouter:

    def __init__(
            self,
            mode: TRoutingMode,
            routing_policies: dict[str, Routable],
            feature_extractor: RAGFeatureExtractor,
        ) -> None:
        self.mode = mode
        self.routing_policies = routing_policies
        self.feature_extractor = feature_extractor

    def select_language_model(self, task_instance: RAGTask) -> TModelSelection:
        fvector = self.feature_extractor.extract_from_task(task_instance)
        match self.mode:

            case "slm":
                selection = "_slm"

            case "llm":
                selection = "_llm"

            case "dynamic-rule-based":
                policy = self.routing_policies[task_instance.name]
                selection = "_llm" if policy.call_large_model(fvector) else "_slm"

            case "dynamic-slm":
                policy = self.routing_policies[task_instance.name]
                selection = "_llm" if policy.call_large_model() else "_slm"

            case _:
                raise ValueError("Unsupported routing mode received: %s", self.mode)
        return fvector, selection
