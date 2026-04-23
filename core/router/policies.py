"""Routing policy implementations for SLM vs LLM selection decisions."""

import logging
import operator
from typing import Protocol, Callable, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException

from core.router.features import TFeatureVector
from core.tasks import RAGTask
from core.messaging import LangchainMessageBuilder

logger = logging.getLogger(__name__)

TOperators = Literal["gt", "ge", "lt", "le", "eq"]

TRoutableFeatures = TFeatureVector | RAGTask

class SLMRouterOutput(BaseModel):
    """Structured output from the SLM routing judge, carrying a Likert-scale LLM-necessity score."""

    confidence: int = Field(
        description=(
            "Integer on a 1–5 Likert scale indicating how strongly the task requires an LLM over an SLM. "
            "1 = trivial for SLM; 3 = borderline; 5 = clearly requires LLM. "
            "Judges the task itself, not the model's self-certainty."
        )
    )


class WeightedRule(BaseModel):
    """A single named threshold rule with an associated weight used in ``WeightedRuleBasedRoutingPolicy``."""

    name: str
    operator: TOperators
    threshold: float
    weight: float


class Routable(Protocol):
    """Structural protocol for routing policies; satisfied by any class with a ``call_large_model`` method."""

    async def call_large_model(self, features: TRoutableFeatures) -> bool:
        """Return ``True`` to route to the large model, ``False`` for the small model."""
        ...


OPERATORS: dict[TOperators, Callable] = {
    "gt":  operator.gt,
    "ge": operator.ge,
    "lt":  operator.lt,
    "le": operator.le,
    "eq": operator.eq,
}


class SLMRoutingPolicy:
    """Delegates the routing decision to an SLM by invoking it with the rendered task prompt.

    Args:
        client: LangChain-wrapped chat model that acts as the routing SLM.
    """
    KEY = "slm_routing_policy"

    def __init__(
            self, 
            client: ChatOpenAI, 
            message_builder: LangchainMessageBuilder,
            confidence_threshold: float
        ) -> None:
        self.client = client
        self.message_builder = message_builder
        self.confidence_threshold = confidence_threshold

    async def call_large_model(self, features: TRoutableFeatures) -> bool:
        """Invoke the SLM with ``features`` and return its routing decision."""
        message = self.message_builder.create_message(
            self.KEY,
            name=features.name,
            query=features.query,
            documents=features.documents
        )
        logger.info("Sending routing request: model=%s", self.client.model_name)
        response: AIMessage = await self.client.ainvoke([message])
        try:
            model: SLMRouterOutput = (
                self.message_builder
                .get_parser(self.KEY)
                .parse(response.content)
            )
            router_model_confidence = model.confidence

        except OutputParserException as ex:
            logger.info("OutputParserException in SLMRoutingPolicy: %s", ex)
            router_model_confidence = 0

        return router_model_confidence >= self.confidence_threshold
        

class WeightedRuleBasedRoutingPolicy:
    """Routes to the LLM when enough weighted feature rules fire simultaneously.

    A rule fires when the named feature value satisfies the rule's operator and threshold.
    The policy triggers LLM routing only if **both** conditions hold:
    - at least ``min_triggers`` rules fire, and
    - the sum of their weights reaches ``cumulative_weights_threshold``.

    Args:
        *rules: ``WeightedRule`` instances defining the routing conditions.
        min_triggers: Minimum number of rules that must fire to consider LLM routing.
        cumulative_weights_threshold: Minimum total weight of triggered rules required.
    """

    def __init__(
        self,
        *rules: WeightedRule,
        min_triggers: int = 1,
        cumulative_weights_threshold: float = 1.0,
    ) -> None:
        self.rules = rules
        self.min_triggers = min_triggers
        self.cumulative_weights_threshold = cumulative_weights_threshold

    async def call_large_model(self, features: TRoutableFeatures) -> bool:
        """Return ``True`` if triggered rules meet both the count and cumulative weight thresholds."""
        feature_values = features.model_dump()
        triggered_weight = 0.0
        triggered_count = 0

        for rule in self.rules:

            if rule.name not in feature_values:
                continue

            if OPERATORS[rule.operator](feature_values[rule.name], rule.threshold):
                triggered_weight += rule.weight
                triggered_count += 1

        return (
            triggered_count >= self.min_triggers
            and triggered_weight >= self.cumulative_weights_threshold
        )
