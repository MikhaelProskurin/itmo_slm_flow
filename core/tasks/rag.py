"""Abstract task interfaces and the concrete RAG task implementation.

Defines how a dataset row is formatted into a prompt, dispatched to an LLM,
and its raw string response returned for downstream evaluation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException

from core.data import DatasetRecord
from core.messaging import LangchainMessageBuilder

class RAGTask:
    
    def __init__(self, name: str, query: str, documents: list[str]) -> None:
        self.name = name
        self.query = query
        self.documents = documents

    @classmethod
    def from_record(cls, record: DatasetRecord) -> "RAGTask":
        return cls(
            record.task, 
            record.sample.query,
            record.sample.documents
        )
    
    async def agenerate_prediction(self, client: ChatOpenAI, messages_builder: LangchainMessageBuilder) -> str:
        message = messages_builder.create_message(
            self.name, 
            query=self.query, 
            documents=self.documents
        )
        response: AIMessage = await client.ainvoke(message)
        try:
            output = messages_builder.get_parser(self.name).parse(response.content)
        except OutputParserException as ex:
            output = ...
        return output
