"""Centralized Pydantic model definitions for the entire slm-flow project.

Defines data shapes for synthetic generation outputs, the unified dataset row,
NLP feature vectors, task I/O, and LLM-as-a-judge evaluation results.
"""

from typing import (
    List,
    Dict,
    Any,
    Optional,
)
from pydantic import BaseModel, Field
from langchain_core.messages.ai import UsageMetadata


class SyntheticDocumentModel(BaseModel):
    """A single document produced during synthetic data generation."""

    idx: int = Field(description="Id of the generated document.", examples=[1, 2, 3])
    content: str = Field(description="Document text.")
    reasoning_trace: Optional[str] = Field(description="The step-by-step reasoning process that led to this answer.")


class SyntheticRowBase(BaseModel):
    """Base fields shared by all synthetic task examples."""

    query: str = Field(description="The user query.")
    documents: List[SyntheticDocumentModel] = Field(description="List of documents with id and full text.")


class SyntheticRowReranking(SyntheticRowBase):
    """Synthetic example for the document reranking task."""

    golden_answer: str = Field(description="The most relevant document")
    sorted_order_ids: Optional[List[int]] = Field(description="Document indexes sorted by document relevance to query.", examples=[[1, 3, 5, 4]])


class SyntheticRowCompression(SyntheticRowBase):
    """Synthetic example for the context compression task."""

    golden_answer: str = Field(description="Compressed context that preserves only the minimal information necessary to answer the question.")
    optimal_compression_length: Optional[int] = Field(description="Length of the optimal compressed context.", examples=[100, 25, 342])


class SyntheticRowDump(BaseModel):
    """Full API response wrapper persisted to disk after each generation call."""

    usage_metadata: Optional[UsageMetadata]
    response_metadata: Optional[Dict[str, Any]]
    task_row_model: SyntheticRowCompression | SyntheticRowReranking | Any


class SlmFlowDatasetRow(BaseModel):
    """Unified dataset row loaded from disk and fed into the inference pipeline."""

    task: str
    domain: str
    difficulty: str
    usage_metadata: Optional[UsageMetadata | Dict[str, Any]]
    response_metadata: Optional[Dict[str, Any]]
    task_row_model: SyntheticRowCompression | SyntheticRowReranking | Any


class RerankingFeatures(BaseModel):
    """Feature vector for reranking routing decision."""

    query_token_count: float
    query_noun_chunk_count: float
    query_avg_word_frequency: float
    avg_lexical_overlap: float
    min_lexical_overlap: float
    inter_document_similarity: float
    documents_count: float


class ContextCompressionFeatures(BaseModel):
    """Feature vector for context compression routing decision."""

    query_token_count: float
    query_noun_chunk_count: float
    query_avg_word_frequency: float
    total_context_token_count: float
    avg_chunk_token_count: float
    avg_lexical_overlap: float
    relevant_documents_ratio: float


class RagTaskOutput(BaseModel):
    """Structured output returned by a RAG task execution."""

    answer: Optional[str] = Field(description="The compressed context or the most relevant document.")


class JudgeVerdict(BaseModel):
    """Structured output from the LLM-as-a-judge evaluator."""

    score: float = Field(description="Quality score from 0 to 10.", examples=[1.0, 5.0, 8.0, 10.0])
    reasoning: str = Field(description="Brief explanation of the score.")


class FlowResult(BaseModel):
    """Single inference + evaluation record produced by InferenceFlow."""

    task: str
    routed_model: str
    query: str
    golden_answer: str
    prediction: str
    features: Optional[RerankingFeatures | ContextCompressionFeatures]
    judge_score: Optional[float]
    judge_reasoning: Optional[str]
