"""Inference pipeline that orchestrates dataset iteration, model routing, and LLM-as-a-judge evaluation.

``InferenceFlow`` iterates over a dataset, delegates each row to the appropriate task,
uses the scheduler to route it to the SLM or LLM, and optionally scores predictions
with an async judge model.
"""

import asyncio
import logging

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser

from pandas import DataFrame

from core.tasks.base import BaseTask
from core.data.datasets import SlmFlowBaseDataset
from core.io.models import JudgeVerdict, FlowResult
from core.scheduler.base import SlmFlowScheduler
from core.flow.utils import (
    run_judge_llm,
    timeit
)

logger = logging.getLogger(__name__)


class InferenceFlow:
    """Main inference pipeline that drives dataset iteration, routing, and evaluation.

    param: dataset: Dataset to iterate over during ``execute``.
       type: SlmFlowBaseDataset
    param: scheduler: Scheduler that selects the model and extracts features for each task.
       type: SlmFlowScheduler
    """

    def __init__(
            self,
            dataset: SlmFlowBaseDataset,
            scheduler: SlmFlowScheduler,
        ) -> None:
        self.dataset = dataset
        self.scheduler = scheduler
        self._to_eval = []
        self._routings = []
        self._evaluation_results = []

    @property
    def evaluation_results_pandas(self) -> DataFrame:
        """Return accumulated evaluation results as a Pandas DataFrame."""
        return DataFrame([r.model_dump() for r in self._evaluation_results])

    @timeit
    def execute(
            self,
            tasks: dict[str, type[BaseTask]],
            parsers: dict[str, PydanticOutputParser],
            templates: dict[str, str]
        ) -> list[str]:
        """Run all dataset rows through the scheduler and return raw model predictions.

        Populates internal buffers (``_to_eval``, ``_routings``) used by
        ``evaluate_by_judge`` for subsequent scoring.

        param: tasks: Map of task name to task class; used to construct a task per row.
           type: dict[str, type[BaseTask]]
        param: parsers: Map of task name to output parser forwarded to each task.
           type: dict[str, PydanticOutputParser]
        param: templates: Map of task name to inference prompt template.
           type: dict[str, str]
        """
        n_total = len(self.dataset)
        logger.info(f"Starting execute: {n_total} rows, mode={self.scheduler.mode}")

        predictions = []
        for i, row in enumerate(self.dataset):

            key, row_model = row.task, row.task_row_model

            task = tasks[key].create(row, parsers[key], templates[key], key)
            model, features = self.scheduler.route(task=task)

            output = task.run(model)

            logger.info(f"[{i + 1}/{n_total}] task={key} routed_to={model.model_name}")

            predictions.append(output)
            self._routings.append(model.model_name)
            self._to_eval.append({
                "task": key,
                "prediction": output,
                "query": row_model.get("query"),
                "golden_answer": row_model.get("golden_answer"),
                "features": features
            })

        logger.info(f"execute done: {len(predictions)} predictions collected")
        return predictions

    async def evaluate_by_judge(self, client: ChatOpenAI, prompt_template: str) -> list[FlowResult]:
        """Score all predictions concurrently using an LLM judge and return structured results.

        param: client: LLM client used for judge evaluation calls.
           type: ChatOpenAI
        param: prompt_template: Evaluation prompt template with ``{query}``, ``{golden_answer}``,
           ``{prediction}``, and ``{fmt}`` placeholders.
           type: str
        """
        n_total = len(self._to_eval)
        logger.info(f"Starting judge evaluation: {n_total} records")

        coroutines = []
        for record in self._to_eval:
            kwargs = {
                "query": record["query"],
                "golden_answer": record["golden_answer"],
                "prediction": record["prediction"]
            }
            coroutines.append(run_judge_llm(client, prompt_template, **kwargs))

        verdicts: list[JudgeVerdict | None] = list(await asyncio.gather(*coroutines))

        failed_verdicts = sum(1 for v in verdicts if v is None)
        if failed_verdicts:
            logger.warning(f"Judge evaluation: {failed_verdicts}/{n_total} verdicts failed to parse")
        logger.info(f"Judge evaluation done: {n_total - failed_verdicts}/{n_total} verdicts collected")

        for record, routed_model, verdict in zip(self._to_eval, self._routings, verdicts):
            self._evaluation_results.append(
                FlowResult(
                    task=record["task"],
                    routed_model=routed_model,
                    query=record["query"],
                    golden_answer=record["golden_answer"],
                    prediction=record["prediction"],
                    features=record["features"],
                    judge_score=verdict.score if verdict else None,
                    judge_reasoning=verdict.reasoning if verdict else None
                )
            )
        return self._evaluation_results
