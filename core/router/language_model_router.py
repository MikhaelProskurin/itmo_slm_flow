"""Language model router that maps each RAG task to either the SLM or the LLM client."""

from typing import Literal

from core.pipeline import TRoutingMode
from core.tasks import RAGTask
from core.router import (
    Routable,
    TFeatureVector,
    RAGFeatureExtractor,
    SLMRoutingPolicy,
    WeightedRuleBasedRoutingPolicy
)

TRoute = Literal["_slm", "_llm"]

TModelSelection = tuple[TFeatureVector, TRoute]


class LMRouter:
    """Routes a ``RAGTask`` to either the small or large language model based on the active mode.

    Supported modes:
        - ``"slm"`` / ``"llm"``: static routing, always selects the respective model.
        - ``"dynamic-rule-based"``: delegates to a ``WeightedRuleBasedRoutingPolicy`` per task.
        - ``"dynamic-slm"``: delegates to an SLM-based ``SLMRoutingPolicy`` per task.

    Args:
        mode: Routing strategy from ``TRoutingMode``.
        routing_policies: Per-task policy objects; required for dynamic modes.
        feature_extractor: Extracts the feature vector used by rule-based policies.
    """

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
        """Compute the feature vector for ``task_instance`` and return ``(fvector, route)``.

        Args:
            task_instance: The RAG task to route.

        Returns:
            A tuple of the extracted feature vector and the routing decision (``"_slm"`` or ``"_llm"``).

        Raises:
            ValueError: If ``self.mode`` is not a recognised routing mode.
        """
        fvector = self.feature_extractor.extract_from_task(task_instance)
        policy = self.routing_policies[task_instance.name]
        
        match self.mode:

            case "slm":
                selection = "_slm"

            case "llm":
                selection = "_llm"

            case "dynamic":

                if isinstance(policy, WeightedRuleBasedRoutingPolicy):
                    use_large_model_decision = policy.call_large_model(fvector)
                
                elif isinstance(policy, SLMRoutingPolicy):
                    use_large_model_decision = policy.call_large_model(task_instance)
                
                selection = "_llm" if use_large_model_decision else "_slm"

            case _:
                raise ValueError("Unsupported routing mode received: %s", self.mode)
        return fvector, selection
