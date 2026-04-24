"""End-to-end RAG inference pipeline: routing, prediction, and LLM-as-a-judge evaluation.

``RAGPipelineRunner`` drives the full experiment loop — iterating over a dataset, routing
each row to the SLM or LLM via ``LMRouter``, gathering predictions concurrently, and
scoring results with BERTScore, ROUGE, and a structured judge model.
"""

import asyncio
import logging
import tiktoken

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from bert_score import score
from rouge import Rouge

from core.tasks import RAGTask, TPredictionResult
from core.data import RAGSyntheticDataset
from core.messaging import LangchainMessageBuilder, TASK_DESCRIPTIONS
from core.router import (
    LMRouter,
    Routable,
    TFeatureVector,
    RAGFeatureExtractor
)
from core.router.language_model_router import TRoutingMode

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JScore(BaseModel):
    """Structured output from the LLM-as-a-judge evaluator."""

    feedback: str = Field(
        description=(
            "Detailed critical analysis covering factual precision, "
            "completeness, and hallucination criteria. "
            "Must be written BEFORE assigning any scores."
        ),
    )
    factual_precision: int = Field(description="How accurately the response conveys facts compared to the reference (1-5).")
    completeness: int = Field(description="How well the response covers all key information from the reference (1-5).")
    hallucination: int = Field(description="Degree to which the response is free of unsupported claims (1-5).")
    final_score: int = Field(description="Minimum of factual_precision, completeness, and hallucination scores.")


class InferenceRecord(BaseModel):
    """Single inference + evaluation record produced by InferenceFlow."""

    task: str
    query: str
    golden_answer: str
    generated_answer: str
    routing: str
    feature_vector: TFeatureVector
    usage_metadata: UsageMetadata


class EvaluationRecord(InferenceRecord):
    """``InferenceRecord`` extended with LLM-as-a-judge scores and automatic answer metrics."""

    jscore: JScore
    answer_metrics: "TAnswerMetrics"


class RerankingMetrics(BaseModel):
    """BERTScore F1 and exact-match hit rate for the reranking task."""

    bert_f1: float
    exact_match: bool


class CompressionMetrics(BaseModel):
    """Quality and compression metrics for the context compression task."""

    bert_f1: float
    rouge_l: float
    rouge_n: float
    compression_ratio: float


TAnswerMetrics = RerankingMetrics | CompressionMetrics


class RAGPipelineRunner:
    """End-to-end pipeline that routes dataset rows to SLM/LLM, runs inference, and evaluates results.

    Manages three ``ChatOpenAI`` clients (small model, large model, judge), a ``LMRouter`` for
    per-row routing decisions, and evaluation logic (BERTScore, ROUGE, LLM-as-a-judge).

    Args:
        small_model: OpenAI model name for the SLM client.
        large_model: OpenAI model name for the LLM client.
        judge_model: OpenAI model name for the evaluator (judge) client.
        routing_mode: One of ``"slm"``, ``"llm"``, ``"dynamic-rule-based"``, ``"dynamic-slm"``.
        messages_builder: Shared builder for rendering task and evaluation prompts.
        dynamic_routing_policies: Per-task ``Routable`` policies; required for dynamic modes.
        extractor_spacy_nlp: spaCy model name passed to ``RAGFeatureExtractor``.
        extractor_tokenizer_name: tiktoken model name passed to ``RAGFeatureExtractor``.
        model_kwargs: Extra keyword arguments are forwarded to every ``ChatOpenAI`` constructor.
    """

    EVALUATION_BUILDER_KEY = "judge"

    def __init__(
            self,
            small_model: str,
            large_model: str,
            judge_model: str,
            routing_mode: TRoutingMode,
            messages_builder: LangchainMessageBuilder,
            dynamic_routing_policies: dict[str, Routable] = None,
            extractor_spacy_nlp: str = None,
            extractor_tokenizer_name: str = None,
            model_kwargs: dict[str, Any] = None
        ) -> None:
        extractor = RAGFeatureExtractor.from_model_names(
            nlp=extractor_spacy_nlp,
            tokenizer=extractor_tokenizer_name
        )
        self.router = LMRouter(routing_mode, dynamic_routing_policies, extractor)
        self.messages_builder = messages_builder
        self._configure_clients(
            (small_model, large_model, judge_model),
            model_kwargs
        )
        self._evaluation_tokenizer = tiktoken.get_encoding("cl100k_base")
        self._rouge = Rouge()

    @property
    def _evaluation_messages_builder_key(self) -> str:
        return self.EVALUATION_BUILDER_KEY

    def _configure_clients(self, names: tuple[str], model_kwargs: dict[str, Any]) -> None:
        """Instantiate SLM, LLM, and judge clients from model name strings."""
        clients = [ChatOpenAI(model=model, **model_kwargs) for model in names]
        self.slm, self.llm, self.judge = tuple(clients)

    def _get_routed_client(self, route: str) -> ChatOpenAI:
        """Return the ``ChatOpenAI`` client corresponding to the routing decision."""
        available_clients = {"_llm": self.llm, "_slm": self.slm}
        return available_clients[route]

    async def arun(self, dataset: RAGSyntheticDataset) -> list[InferenceRecord]:
        """Run inference over the entire dataset, routing each row to SLM or LLM.

        All prediction coroutines are gathered concurrently after routing decisions are made
        synchronously for the full dataset.

        Args:
            dataset: Loaded synthetic dataset to iterate over.

        Returns:
            List of ``InferenceRecord`` objects, one per dataset row.
        """
        logger.info("arun started: dataset_size=%d, routing_mode=%s", len(dataset), self.router.mode)

        tis = [RAGTask.from_record(row) for row in dataset.rows]

        routing_results = await asyncio.gather(
            *[self.router.select_language_model(ti) for ti in tis]
        )

        answer_generating_coroutines = [
            ti.agenerate_prediction(self._get_routed_client(route), self.messages_builder)
            for ti, (_, route) in zip(tis, routing_results)
        ]
        generated_answers: list[TPredictionResult] = await asyncio.gather(*answer_generating_coroutines)

        records = [
            InferenceRecord(
                task=ti.name,
                query=ti.query,
                golden_answer=row.sample.golden_answer,
                generated_answer=answer.content,
                routing=route,
                feature_vector=fvector,
                usage_metadata=usage,
            )
            for ti, (fvector, route), row, (answer, usage)
            in zip(tis, routing_results, dataset.rows, generated_answers)
        ]

        logger.info("arun finished: records_produced=%d", len(records))
        return records

    async def aevaluate(self, generated_answers: list[InferenceRecord]) -> list[EvaluationRecord]:
        """Score a list of inference records with the judge model and automatic metrics.

        Computes task-specific metrics (BERTScore, ROUGE, compression ratio) and concurrently
        invokes the LLM judge for all records.

        Args:
            generated_answers: Records produced by ``arun``.

        Returns:
            List of ``EvaluationRecord`` objects with jscores and answer metrics populated.

        Raises:
            ValueError: If a record contains an unsupported task name.
        """
        logger.info("aevaluate started: records_count=%d", len(generated_answers))

        key = self.EVALUATION_BUILDER_KEY

        messages, computed_metrics = [], []
        for record in generated_answers:

            task = record.task
            description = TASK_DESCRIPTIONS[task]

            messages.append(
                self.messages_builder.create_message(
                    key,
                    task_type=task,
                    task_description=description,
                    query=record.query,
                    prediction=record.generated_answer,
                    golden_answer=record.golden_answer
                )
            )

            match task:

                case "reranking":
                    metrics = self._compute_reranking_metrics(record)

                case "context_compression":
                    metrics = self._compute_context_compression_metrics(record)

                case _:
                    raise ValueError("Unsupported metrics computation for task: %s", task)

            computed_metrics.append(metrics)

        p = self.messages_builder.get_parser(key)
        jscore_coroutines = [self._agenerate_jscore(m, p) for m in messages]
        jscores = await asyncio.gather(*jscore_coroutines)

        results = [
            EvaluationRecord(
                **model.model_dump(),
                jscore=jsc,
                answer_metrics=metrics
            )
            for model, jsc, metrics
            in zip(generated_answers, jscores, computed_metrics)
        ]

        logger.info("aevaluate finished: records_evaluated=%d", len(results))
        return results

    async def _agenerate_jscore(
            self,
            input_message: SystemMessage,
            output_parser: PydanticOutputParser[JScore]
        ) -> JScore:
        """Invoke the judge model and parse its structured evaluation.

        Returns:
            Parsed ``JScore``, or a sentinel ``JScore`` with all scores set to ``0`` and
            ``feedback="structured_output_parsing_error"`` on ``OutputParserException``.
        """
        logger.info("Sending judge evaluation request: model=%s", self.judge.model_name)
        response: AIMessage = await self.judge.ainvoke([input_message])
        try:
            jscore = output_parser.parse(response.content)
        except OutputParserException as ex:
            logger.info("OutputParserException in judge evaluation: %s", ex)
            jscore = JScore(
                feedback="structured_output_parsing_error",
                factual_precision=0,
                completeness=0,
                hallucination=0,
                final_score=0
            )
        return jscore

    @staticmethod
    def _compute_reranking_metrics(record: InferenceRecord) -> RerankingMetrics:
        """Compute BERTScore F1 and exact-match for a reranking prediction."""
        candidate, reference = record.generated_answer, record.golden_answer
        hit = candidate.strip() == reference.strip()

        _precision, _recall, f1 = score(
            cands=[candidate],
            refs=[reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
        return RerankingMetrics(
            bert_f1=f1.item(),
            exact_match=hit
        )

    def _compute_context_compression_metrics(self, record: InferenceRecord) -> CompressionMetrics:
        """Compute BERTScore, ROUGE-L/2, and token compression ratio for a compression prediction."""
        candidate, reference = record.generated_answer, record.golden_answer

        _precision, _recall, f1 = score(
            cands=[candidate],
            refs=[reference],
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
        )
        rouge_score = self._rouge.get_scores(
            hyps=[candidate], refs=[reference], avg=False
        )[0]
        compression_ratio = (
            len(self._evaluation_tokenizer.encode(candidate)) / record.feature_vector.total_context_token_count
        )
        return CompressionMetrics(
            bert_f1=f1.item(),
            rouge_l=rouge_score["rouge-l"]["f"],
            rouge_n=rouge_score["rouge-2"]["f"],
            compression_ratio=compression_ratio
        )
