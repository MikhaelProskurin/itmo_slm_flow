"""Routing policies that decide whether a request should go to the SLM or LLM.

Each policy implements ``BaseRoutingPolicy`` and returns ``True`` to escalate to
the LLM, or ``False`` to keep it with the SLM.
"""

import logging
import numpy as np
import operator as op

from typing import Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

from core.io.models import RerankingFeatures, ContextCompressionFeatures


FeatureVector = RerankingFeatures | ContextCompressionFeatures

_OPERATORS: dict[str, Any] = {
    "gt":  op.gt,
    "ge": op.ge,
    "lt":  op.lt,
    "le": op.le,
    "eq": op.eq,
}


class BaseRoutingPolicy(ABC):
    """Base class for SLM/LLM routing policies."""

    @abstractmethod
    def decide(self, features: FeatureVector) -> bool:
        """Return ``True`` to route to LLM, ``False`` to route to SLM."""
        pass


class ThresholdRoutingPolicy(BaseRoutingPolicy):
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

        logger.debug("ThresholdRoutingPolicy.decide | features: %s", feature_values)

        for feature_name, (operator, threshold, weight) in self.rules.items():
            if feature_name not in feature_values:
                continue

            if _OPERATORS[operator](feature_values[feature_name], threshold):
                triggered_weight += weight
                triggered_count += 1

        return triggered_count >= self.min_triggers and triggered_weight >= self.total_threshold


class MLRoutingPolicy(BaseRoutingPolicy):
    """Sklearn-based routing policy trained on labeled feature vectors.

    param: model: Fitted sklearn classifier with ``predict_proba`` support.
       type: Any
    param: llm_threshold: Minimum predicted P(llm) required to route to LLM.
       type: float

    Example::

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression().fit(X_train, y_train)

        policy = MLRoutingPolicy(model=clf, llm_threshold=0.6)
        use_llm = policy.decide(features)  # True if P(llm) >= 0.6
    """

    def __init__(self, model: Any, llm_threshold: float = 0.5) -> None:
        self.model = model
        self.llm_threshold = llm_threshold

    def decide(self, features: FeatureVector) -> bool:
        """Return ``True`` if the classifier's LLM probability meets ``llm_threshold``."""
        feature_dump = features.model_dump()
        logger.debug("MLRoutingPolicy.decide | features: %s", feature_dump)
        vector = np.array([[feature_dump[f] for f in feature_dump]])
        proba = self.model.predict_proba(vector)[0][1]
        return proba >= self.llm_threshold
