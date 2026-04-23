"""Feature extraction for RAG routing decisions using spaCy NLP and tiktoken tokenisation."""

import spacy
from spacy.tokens import Doc, Token

import tiktoken
from wordfreq import word_frequency

from pydantic import BaseModel

from core.tasks import RAGTask


class RAGFeatureVectorBase(BaseModel):
    """Shared query-level features common to all RAG task feature vectors."""

    query_token_count: float
    query_noun_chunk_count: float
    query_avg_word_frequency: float
    avg_lexical_overlap: float


class RerankingVector(RAGFeatureVectorBase):
    """Feature vector for reranking routing decision."""

    min_lexical_overlap: float
    documents_cosine_similarity: float
    documents_count: float


class CompressionVector(RAGFeatureVectorBase):
    """Feature vector for context compression routing decision."""

    total_context_token_count: float
    avg_chunk_token_count: float
    relevant_documents_ratio: float


TFeatureVector = RerankingVector | CompressionVector
TQueryFeatures = tuple[float, float, float]

class RAGFeatureExtractor:
    """Extracts numeric feature vectors from ``RAGTask`` instances for routing policy evaluation.

    Uses a spaCy pipeline for NLP features (noun chunks, lemmas, cosine similarity) and a
    tiktoken encoder for token counts. Dispatches to task-specific computation methods based
    on the task name.

    Args:
        nlp: Loaded spaCy language model.
        tokenizer: tiktoken encoding used for token counting.
    """

    RELEVANCE_THRESHOLD = 0.3

    def __init__(self, nlp: spacy.Language, tokenizer: tiktoken.Encoding) -> None:
        self.nlp = nlp
        self.tokenizer = tokenizer

    @property
    def relevance_threshold(self) -> float:
        """Lexical overlap threshold above which a document is considered relevant."""
        return self.RELEVANCE_THRESHOLD

    @classmethod
    def from_model_names(cls, nlp: str, tokenizer: str) -> "RAGFeatureExtractor":
        """Construct by loading spaCy and tiktoken models by name."""
        return cls(
            nlp=spacy.load(nlp),
            tokenizer=tiktoken.get_encoding(tokenizer)
        )

    def extract_from_task(self, task: RAGTask) -> TFeatureVector:
        """Dispatch feature extraction to the appropriate method based on ``task.name``.

        Args:
            task: The RAG task instance to featurise.

        Returns:
            A ``RerankingVector`` or ``CompressionVector`` depending on the task type.

        Raises:
            ValueError: If the task name is not ``"reranking"`` or ``"context_compression"``.
        """
        tname, query, documents = task.name, task.query, task.documents
        match tname:
            case "reranking":
                fvector = self.compute_reranking_feature_vector(query, documents)
            case "context_compression":
                fvector = self.compute_compression_feature_vector(query, documents)
            case _:
                raise ValueError("Unsupported feature extraction for task: %s", tname)
        return fvector

    def compute_reranking_feature_vector(self, query: str, documents: list[str]) -> RerankingVector:
        """Compute the full feature vector for a reranking task instance."""
        query_token_count, query_noun_chunk_count, query_avg_word_frequency = self.get_query_features(query)
        lexical_overlaps = self.get_sample_vocabulary_intersection(query, documents)

        return RerankingVector(
            query_token_count=query_token_count,
            query_noun_chunk_count=query_noun_chunk_count,
            query_avg_word_frequency=query_avg_word_frequency,
            avg_lexical_overlap=sum(lexical_overlaps) / len(lexical_overlaps),
            min_lexical_overlap=min(lexical_overlaps),
            documents_cosine_similarity=self.get_documents_avg_cosine_similarity(documents),
            documents_count=len(documents)
        )

    def compute_compression_feature_vector(self, query: str, documents: list[str]) -> CompressionVector:
        """Compute the full feature vector for a context compression task instance."""
        query_token_count, query_noun_chunk_count, query_avg_word_frequency = self.get_query_features(query)
        lexical_overlaps = self.get_sample_vocabulary_intersection(query, documents)

        tokens_by_document = [len(self.tokenizer.encode(d)) for d in documents]
        relevant_documents_count = sum(1 for overlap in lexical_overlaps if overlap >= self.RELEVANCE_THRESHOLD)

        return CompressionVector(
            query_token_count=query_token_count,
            query_noun_chunk_count=query_noun_chunk_count,
            query_avg_word_frequency=query_avg_word_frequency,
            avg_lexical_overlap=sum(lexical_overlaps) / len(lexical_overlaps),
            total_context_token_count=sum(tokens_by_document),
            avg_chunk_token_count=sum(tokens_by_document) / len(tokens_by_document),
            relevant_documents_ratio=relevant_documents_count / len(documents)
        )

    def get_documents_avg_cosine_similarity(self, documents: list[str]) -> float:
        """Return the mean pairwise spaCy cosine similarity across all document pairs."""
        serialized_documents: list[Doc] = list(self.nlp.pipe(documents, n_process=1))

        similarities = []
        for i in range(len(serialized_documents)):
            for j in range(i + 1, len(serialized_documents)):
                similarities.append(
                    serialized_documents[i].similarity(serialized_documents[j])
                )

        return sum(similarities) / len(similarities)

    def get_sample_vocabulary_intersection(self, query: str, documents: list[str]) -> list[float]:
        """Return per-document lexical overlap ratios between query lemmas and document lemmas."""
        serialized_query = self.serialize_to_spacy(query)
        query_terms = self._to_unique_lemmas(serialized_query)

        intersections = []
        for document in documents:
            serialized_document = self.serialize_to_spacy(document)
            document_terms = self._to_unique_lemmas(serialized_document)

            overlap = len(query_terms & document_terms) / max(len(query_terms), 1)
            intersections.append(overlap)

        return intersections

    def get_query_features(self, query: str) -> TQueryFeatures:
        """Return ``(token_count, noun_chunk_count, avg_word_frequency)`` for the query."""
        spacy_document = self.serialize_to_spacy(query)

        num_tokens = len(self.tokenizer.encode(query))

        words = [token.text for token in spacy_document]
        word_frequencies = [word_frequency(w, lang="en") for w in words]
        avg_word_frequency = sum(word_frequencies) / len(words)

        noun_chunks_count = len(list(self.nlp(query).noun_chunks))

        return num_tokens, noun_chunks_count, avg_word_frequency

    def serialize_to_spacy(self, document: str) -> list[Token]:
        """Lowercase, parse with spaCy, and return only content tokens (alpha, non-stop, non-punct)."""
        spacy_document = self.nlp(document.lower())
        clear_tokens = [token for token in spacy_document if self._is_clear_token(token)]
        return clear_tokens

    @staticmethod
    def _to_unique_lemmas(tokens: list[Token]) -> set[str]:
        return {token.lemma_ for token in tokens}

    @staticmethod
    def _is_clear_token(token: Token) -> bool:
        return token.is_alpha and not (token.is_punct or token.is_stop or token.is_space)
