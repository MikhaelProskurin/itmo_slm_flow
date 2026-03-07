"""Central scheduler that wires together feature extractors and routing policies.

``SlmFlowScheduler`` selects the appropriate model client (SLM or LLM) for each
incoming task based on extracted features and the active operating mode.
"""

import logging
from typing import Literal

from langchain_openai import ChatOpenAI

from core.io.models import RerankingFeatures, ContextCompressionFeatures
from core.scheduler.features import BaseFeatureExtractor
from core.scheduler.policies import BaseRoutingPolicy
from core.tasks.base import RagTask


logger = logging.getLogger(__name__)

TaskInput = RagTask


class SlmFlowScheduler:
    """Routes each task to either the SLM or LLM based on extracted features.

    param: policies: Map of task name to its routing policy; used only in ``"dynamic"`` mode.
       type: dict[str, BaseRoutingPolicy]
    param: extractors: Map of task name to its feature extractor.
       type: dict[str, BaseFeatureExtractor]
    param: llm: Large language model client, selected when policy returns ``True``.
       type: ChatOpenAI
    param: slm: Small language model client, selected when policy returns ``False``.
       type: ChatOpenAI
    param: mode: Operating mode — ``"dynamic"`` uses the policy; ``"llm_only"`` / ``"slm_only"``
       always route to a fixed model regardless of features.
       type: Literal["llm_only", "slm_only", "dynamic"]
    """

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
        """Return the selected model client and the extracted feature vector for ``task``.

        Features are always extracted regardless of the operating mode so they can
        be recorded in ``FlowResult`` for later analysis.

        param: task: Incoming task whose ``task_name`` is used to look up extractor and policy.
           type: RagTask
        """
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
