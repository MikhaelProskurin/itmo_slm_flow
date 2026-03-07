"""Feature extractors that transform RagTask inputs into numeric feature vectors for routing.

Provides task-specific extractors (reranking, context compression) that share a
common NLP pipeline (spaCy + tiktoken) via an abstract base class.
"""

from abc import ABC, abstractmethod

import spacy
import tiktoken

from core.io.models import (
    RerankingFeatures,
    ContextCompressionFeatures,
)
from core.tasks.base import RagTask
from core.scheduler.utils import query_features, lexical_overlap


class BaseFeatureExtractor(ABC):
    """Abstract base for all feature extractors; initializes the shared NLP pipeline.

    param: spacy_model: spaCy model name loaded for NLP processing.
       type: str
    param: tokenizer_model: tiktoken-compatible model name used for token counting.
       type: str
    """

    def __init__(
            self,
            spacy_model: str = "en_core_web_lg",
            tokenizer_model: str = "gpt-4o",
        ) -> None:
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)

    @abstractmethod
    def extract(self, task) -> RerankingFeatures | ContextCompressionFeatures:
        """Extract a feature vector from ``task`` for routing decision."""
        pass


class RerankingFeatureExtractor(BaseFeatureExtractor):
    """Extracts routing features for the reranking task."""

    def extract(self, task: RagTask) -> RerankingFeatures:
        """Compute query-level stats, per-document lexical overlaps, and inter-document similarity.

        param: task: Populated RagTask instance containing query and documents.
           type: RagTask
        """
        query_feats = query_features(task.query, self.nlp, self.tokenizer)
        overlaps = [lexical_overlap(self.nlp, task.query, doc.get("content", "")) for doc in task.documents]
        inter_sim = self._inter_document_similarity(task.documents)

        return RerankingFeatures(
            **query_feats,
            avg_lexical_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
            min_lexical_overlap=min(overlaps) if overlaps else 0.0,
            inter_document_similarity=inter_sim,
            documents_count=float(len(task.documents)),
        )

    def _inter_document_similarity(self, documents: list[str]) -> float:
        """Compute average pairwise cosine similarity between documents via spacy vectors."""
        if len(documents) < 2:
            return 0.0

        docs = list(self.nlp.pipe(doc.get("content", "") for doc in documents))
        pairs = [
            docs[i].similarity(docs[j])
            for i in range(len(docs))
            for j in range(i + 1, len(docs))
        ]
        return sum(pairs) / len(pairs)


class ContextCompressionFeatureExtractor(BaseFeatureExtractor):
    """Extracts routing features for the context compression task."""

    RELEVANCE_THRESHOLD = 0.3

    def extract(self, task: RagTask) -> ContextCompressionFeatures:
        """Compute query stats, chunk token counts, lexical overlaps, and relevant-document ratio.

        param: task: Populated RagTask instance containing query and documents.
           type: RagTask
        """
        query_feats = query_features(task.query, self.nlp, self.tokenizer)

        chunk_token_counts = [len(self.tokenizer.encode(doc.get("content", ""))) for doc in task.documents]
        overlaps = [lexical_overlap(self.nlp, task.query, doc.get("content", "")) for doc in task.documents]
        
        relevant_count = sum(1 for o in overlaps if o >= self.RELEVANCE_THRESHOLD)

        return ContextCompressionFeatures(
            **query_feats,
            total_context_token_count=sum(chunk_token_counts),
            avg_chunk_token_count=sum(chunk_token_counts) / len(chunk_token_counts) if chunk_token_counts else 0.0,
            avg_lexical_overlap=sum(overlaps) / len(overlaps) if overlaps else 0.0,
            relevant_documents_ratio=relevant_count / len(task.documents) if task.documents else 0.0,
        )
