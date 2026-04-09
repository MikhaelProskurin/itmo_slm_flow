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
        task_row = row.task_row_model
        if isinstance(task_row, BaseModel):
            query = task_row.query
            documents = task_row.documents
        else:
            query = task_row.get("query")
            documents = task_row.get("documents")
        return cls(task_name, query, documents, parser, prompt_template)

    def run(self, client: ChatOpenAI) -> str:
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
