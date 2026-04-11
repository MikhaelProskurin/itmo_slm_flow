"""Inference pipeline that orchestrates dataset iteration, model routing, and LLM-as-a-judge evaluation.

``InferenceFlow`` iterates over a dataset, delegates each row to the appropriate task,
uses the scheduler to route it to the SLM or LLM, and optionally scores predictions
with an async judge model.
"""

import asyncio

from typing import Literal, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from pandas import DataFrame

from core.tasks.rag import RAGTask

from core.data import RAGSyntheticDataset
from core.messaging import LangchainMessageBuilder
from core.router import (
    TRoute,
    LMRouter,
    Routable,
    TFeatureVector,
    RAGFeatureExtractor
)

from pydantic import BaseModel, Field

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


class EvaluationRecord(BaseModel):
    jscore: JScore
    slm_routing_ratio: float
    slm_success_ratio: float
    answer_metrics: "TAnswerMetrics"


class RerankingMetrics(BaseModel):
    bert_f1: float


class CompressionMetrics(BaseModel):
    bert_f1: float
    rouge_n: float
    compression_ratio: float

TAnswerMetrics = RerankingMetrics | CompressionMetrics
TRoutingMode = Literal["slm", "llm", "dynamic"]

class RAGPipelineRunner:
    
    def __init__(
            self,
            small_model: str,
            large_model: str,
            judge_model: str, 
            routing_mode: TRoutingMode, 
            messages_builder: LangchainMessageBuilder, 
            dynamic_routing_policiy: Routable = None,
            extractor_spacy_nlp: str = None,
            extractor_tokenizer_name: str = None,
            model_kwargs: dict[str, Any] = None
        ) -> None:
        extractor = RAGFeatureExtractor.from_model_names(
            nlp=extractor_spacy_nlp,
            tokenizer=extractor_tokenizer_name
        )
        self.router = LMRouter(routing_mode, dynamic_routing_policiy, extractor)
        self.messages_builder = messages_builder
        self._configure_clients(
            (small_model, large_model, judge_model),
            model_kwargs
        )


    def _configure_clients(self, names: tuple[str], model_kwargs: dict[str, Any]) -> None:
        clients = [ChatOpenAI(model, **model_kwargs) for model in names]
        self.slm, self.llm, self.judge = tuple(clients)
    

    def _get_routed_client(self, route: TRoute) -> ChatOpenAI:
        available_clients = {"_llm": self.llm, "_slm": self.slm}
        return available_clients[route]


    async def arun(self, dataset: RAGSyntheticDataset) -> list[InferenceRecord]:

        tis, routes, features, coroutines = [], [], [], []

        for row in dataset:
            
            ti = RAGTask.from_record(row)
            fvector, route = self.router.select_language_model(ti)

            coroutine = ti.agenerate_prediction(
                self._get_routed_client(route), 
                self.messages_builder
            )

            tis.append(ti)
            routes.append(route)
            features.append(fvector)
            coroutines.append(coroutine)

        generated_answers = asyncio.gather(*coroutines)

        records = [
            InferenceRecord(
                task=ti.name,
                query=ti.query,
                golden_answer=row.sample.golden_answer,
                generated_answer=answer,
                routing=route,
                feature_vector=fvector
            )
            for ti, route, fvector, row, answer
            in zip(tis, routes, features, dataset.rows, generated_answers)
        ]
        return records


    async def aevaluate(self, generated_answers: list[InferenceRecord]) -> list[EvaluationRecord]:
        ...


    def score_routing(self):
        ...


    def score_answer(self):
        ...


    def _compute_reranking_metrics(self) -> RerankingMetrics:
        ...
    

    def _compute_context_compression_metrics(self) -> CompressionMetrics:
        ...
