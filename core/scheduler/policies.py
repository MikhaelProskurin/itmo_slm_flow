"""Routing policies that decide whether a request should go to the SLM or LLM.

Each policy implements ``BaseRoutingPolicy`` and returns ``True`` to escalate to
the LLM, or ``False`` to keep it with the SLM.
"""

import numpy as np
import operator as op

from typing import Any
from abc import ABC, abstractmethod

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
    """Routes to LLM when any feature satisfies its threshold rule.

    Rules are evaluated with OR semantics: a single satisfied condition
    is enough to escalate to LLM, erring on the side of quality.

    param: rules: Mapping of feature name to ``(operator, threshold)`` pairs.
       Supported operators: ``"gt"``, ``"ge"``, ``"lt"``, ``"le"``, ``"eq"``.
       type: dict[str, tuple[str, float]]

    Example::

        policy = ThresholdRoutingPolicy({
            "query_token_count":   ("lt",  500.0),
            "avg_lexical_overlap": ("gt",  0.1),
        })
        use_llm = policy.decide(features)  # True if either rule is satisfied
    """

    def __init__(self, rules: dict[str, tuple[str, float]]) -> None:
        self.rules = rules

    def decide(self, features: FeatureVector) -> bool:
        """Return ``True`` if any rule condition is met (route to LLM)."""
        feature_values = features.model_dump()
        for feature_name, (operator, threshold) in self.rules.items():

            if _OPERATORS[operator](feature_values[feature_name], threshold):
                return True
        return False


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
        vector = np.array([[feature_dump[f] for f in feature_dump]])
        proba = self.model.predict_proba(vector)[0][1]
        return proba >= self.llm_threshold
