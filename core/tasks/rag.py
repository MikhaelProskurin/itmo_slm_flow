"""RAG task abstraction wrapping a single inference request (query + documents → prediction)."""

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException

from core.data import DatasetRecord
from core.messaging import LangchainMessageBuilder


class RAGTaskPrediction(BaseModel):
    """Structured output from a RAG inference call, holding the model's raw text prediction."""

    content: str = Field(description="Model's task output: top-ranked document or compressed context.")


class RAGTask:
    """Encapsulates a single RAG inference unit: a named task, a query, and its candidate documents.

    Args:
        name: Task identifier (e.g. ``"reranking"`` or ``"context_compression"``).
        query: The user query string.
        documents: Plain-text document contents for the task context.
    """

    def __init__(self, name: str, query: str, documents: list[str]) -> None:
        self.name = name
        self.query = query
        self.documents = documents

    @classmethod
    def from_record(cls, record: DatasetRecord) -> "RAGTask":
        """Construct a ``RAGTask`` from a loaded ``DatasetRecord``."""
        return cls(
            record.task,
            record.sample.query,
            [d.content for d in record.sample.documents]
        )

    async def agenerate_prediction(self, client: ChatOpenAI, messages_builder: LangchainMessageBuilder) -> RAGTaskPrediction:
        """Invoke the model and return the parsed prediction string.

        Args:
            client: LangChain-wrapped chat model to call.
            messages_builder: Builder used to render the task prompt and retrieve the output parser.

        Returns:
            Parsed prediction string, or ``"structured_output_parsing_error"`` on ``OutputParserException``.
        """
        message = messages_builder.create_message(
            self.name,
            query=self.query,
            documents=self.documents
        )
        response: AIMessage = await client.ainvoke(message)
        try:
            output = messages_builder.get_parser(self.name).parse(response.content)
        except OutputParserException:
            output = RAGTaskPrediction(content="structured_output_parsing_error")
        return output
