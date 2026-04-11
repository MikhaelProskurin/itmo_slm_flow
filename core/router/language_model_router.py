"""Central scheduler that wires together feature extractors and routing policies.

``SlmFlowScheduler`` selects the appropriate model client (SLM or LLM) for each
incoming task based on extracted features and the active operating mode.
"""

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
            routing_policy: Routable, 
            feature_extractor: RAGFeatureExtractor,
        ) -> None:
        self.mode = mode
        self.routing_policy = routing_policy
        self.feature_extractor = feature_extractor

    def select_language_model(self, task_instance: RAGTask) -> TModelSelection:
        fvector = self.feature_extractor.extract_from_task(task_instance)
        match self.mode:

            case "slm":
                selection = "_slm"
            case "llm":
                selection = "_llm"
            case "dynamic":
                selection = "_llm" if self.routing_policy.call_large_model(fvector) else "_slm"
            case _:
                raise ValueError("Unsupported routing mode recieved: %s", self.mode)
        return fvector, selection
