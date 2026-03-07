"""Abstract task interfaces and the concrete RAG task implementation.

Defines how a dataset row is formatted into a prompt, dispatched to an LLM,
and its raw string response returned for downstream evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from core.io.models import SlmFlowDatasetRow


class BaseTask(ABC):
    """Abstract base class defining the task interface consumed by ``InferenceFlow``.

    param: task_name: Identifier matching the task key in scheduler/parser/template dicts.
       type: str
    param: query: User query extracted from the dataset row.
       type: str
    param: documents: List of retrieved documents (Pydantic models or raw dicts).
       type: list[BaseModel | str]
    param: parser: Pydantic parser used to validate the LLM response format.
       type: PydanticOutputParser
    param: prompt_template: Inference prompt template with ``{query}``, ``{documents}``, ``{fmt}`` placeholders.
       type: str
    """

    def __init__(
        self,
        task_name: str,
        query: str,
        documents: list[BaseModel | dict[str, str]],
        parser: PydanticOutputParser,
        prompt_template: str,
    ) -> None:
        self.task_name = task_name
        self.query = query
        self.documents = documents
        self.parser = parser
        self.prompt_template = prompt_template

    @abstractmethod
    def run(self, client: ChatOpenAI) -> str:
        """Execute the task against ``client`` and return the raw model response string."""
        pass

    @classmethod
    @abstractmethod
    def create(
        cls,
        row: SlmFlowDatasetRow,
        parser: PydanticOutputParser,
        prompt_template: str,
        task_name: str,
    ) -> BaseTask:
        """Construct a task instance from a dataset row."""
        pass


class RagTask(BaseTask):
    """Concrete RAG task that formats a prompt from a dataset row and invokes the LLM synchronously."""

    @classmethod
    def create(cls, row: SlmFlowDatasetRow, parser: PydanticOutputParser, prompt_template: str, task_name: str) -> RagTask:
        """Build a ``RagTask`` by extracting ``query`` and ``documents`` from ``row``.

        param: row: Loaded dataset row supplying query and documents.
           type: SlmFlowDatasetRow
        param: parser: Output parser forwarded to the task instance.
           type: PydanticOutputParser
        param: prompt_template: Inference prompt template forwarded to the task instance.
           type: str
        param: task_name: Task identifier forwarded to the task instance.
           type: str
        """
        task_row = row.task_row_model
        if isinstance(task_row, BaseModel):
            query = task_row.query
            documents = task_row.documents
        else:
            query = task_row.get("query")
            documents = task_row.get("documents")
        return cls(task_name, query, documents, parser, prompt_template)

    def run(self, client: ChatOpenAI) -> str:
        """Format the prompt, call ``client``, and return the raw response content.

        ``OutputParserException`` is silently suppressed — the raw string is always returned
        so the flow can still record and evaluate partial or malformed outputs.

        param: client: LLM client (SLM or LLM) selected by the scheduler.
            type: ChatOpenAI
        """
        kwargs = {
            "query": self.query,
            "documents": self.documents,
            "fmt": self.parser.get_format_instructions(),
        }
        prompt = [SystemMessage(self.prompt_template.format(**kwargs))]
        model_response: AIMessage = client.invoke(prompt)
        try:
            self.parser.parse(model_response.content)
        except OutputParserException:
            pass
        return model_response.content
