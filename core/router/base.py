"""Central scheduler that wires together feature extractors and routing policies.

``SlmFlowScheduler`` selects the appropriate model client (SLM or LLM) for each
incoming task based on extracted features and the active operating mode.
"""

import logging
from typing import Literal

from langchain_openai import ChatOpenAI

from core.io.models import RerankingFeatures, ContextCompressionFeatures
from core.router.features import BaseFeatureExtractor
from core.router.policies import BaseRoutingPolicy
from core.tasks.base import RagTask


logger = logging.getLogger(__name__)

TaskInput = RagTask


class SlmFlowScheduler:

    def __init__(
        self,
        policies: dict[str, BaseRoutingPolicy],
        extractors: dict[str, BaseFeatureExtractor],
        llm: ChatOpenAI,
        slm: ChatOpenAI,
        mode: Literal["llm_only", "slm_only", "dynamic"] = "dynamic"
    ) -> None:
        self.policies = policies
        self.extractors = extractors
        self.llm = llm
        self.slm = slm
        self.mode = mode


    def route(self, task: TaskInput) -> tuple[ChatOpenAI, RerankingFeatures | ContextCompressionFeatures]:

        features = self.extractors[task.task_name].extract(task)

        match self.mode:
            case "llm_only":
                model = self.llm
            case "slm_only":
                model = self.slm
            case "dynamic":
                use_llm = self.policies[task.task_name].decide(features)
                model = self.llm if use_llm else self.slm
            case _:
                raise ValueError(f"Invalid mode [{self.mode}] passed to scheduler!")

        logger.info("[%s] routed to %s", task.task_name, model.model_name)
        return model, features
