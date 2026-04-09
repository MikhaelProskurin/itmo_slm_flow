"""Synthetic dataset generation using LLM APIs.

Provides a configurable, async generator that iterates over task, domain, and
difficulty combinations to produce structured training examples via
``langchain``-based prompt chains with automatic parsing and fallback logic.
"""

import uuid
import asyncio
import aiofiles
import logging

from typing import Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from core.messaging import LangchainMessageBuilder

logger = logging.getLogger(__name__)

TCombo = dict[tuple[str, str, str], int]


class RAGDocument(BaseModel):
    """A single document produced during synthetic data generation."""

    idx: int = Field(description="Id of the generated document.", examples=[1, 2, 3])
    content: str = Field(description="Document text.")
    reasoning_trace: str | None = Field(description="The step-by-step reasoning process that led to this answer.")


class RAGSampleBase(BaseModel):
    """Base fields shared by all synthetic task examples."""

    query: str = Field(description="The user query.")
    documents: list[RAGDocument] = Field(description="List of documents with id and full text.")


class RerankingSample(RAGSampleBase):
    """Synthetic example for the document reranking task."""

    golden_answer: str = Field(description="The most relevant document")


class CompressionSample(RAGSampleBase):
    """Synthetic example for the context compression task."""

    golden_answer: str = Field(description="Compressed context that preserves only the minimal information necessary to answer the question.")


class PersistentSample(BaseModel):
    """Full API response wrapper persisted to disk after each generation call."""

    usage_metadata: UsageMetadata | None
    sample: CompressionSample | RerankingSample


class DatasetDeclaration(BaseModel):
    
    tasks: Sequence[str]
    domains: Sequence[str]
    difficulties: Sequence[str]
    batch_size: int = 10

    @property
    def n_samples(self) -> int:
        """Total number of examples that will be generated across all combinations."""
        return len(self.tasks) * len(self.domains) * len(self.difficulties) * self.batch_size


class RAGDatasetAsyncGenerator:
    
    def __init__(
        self, 
        client: ChatOpenAI, 
        declaration: DatasetDeclaration,
        messages_builder: LangchainMessageBuilder, 
        rate_limit: int = 15
    ) -> None:
        self.client = client
        self.declaration = declaration
        self.messages_builder = messages_builder
        self.rate_limit = rate_limit
    
    async def agenerate_dataset(self, output_dir: str = "./tmp") -> list:
        combos = {
            (task, domain, difficulty): 0
            for task in self.declaration.tasks
            for domain in self.declaration.domains
            for difficulty in self.declaration.difficulties
        }
        messages = [
            self.messages_builder.create_message(
                key, 
                domain=domain,
                difficulty=difficulty
            ) 
            for key, domain, difficulty in combos    
        ]
        output_parsers = [self.messages_builder.get_parser(key) for key, _, _ in combos]

        results = []
        while future_samples := self.get_remaining_samples(combos):

            coroutines = []
            for key, m, p in zip(future_samples, messages, output_parsers):
                
                coroutine = self.agenerate_batch(m, p, batch_size=future_samples[key])
                coroutines.append(coroutine)

            batches = await asyncio.gather(*coroutines)
            
            for key, batch in zip(future_samples, batches):
                combos[key] += len(batch)
                
                path = Path(output_dir) / Path(*key)
                persistence_coroutines = [self.apersist_sample(sample, path) for sample in batch]
                await asyncio.gather(*persistence_coroutines)
            
                results.extend(batch)
            
        return results


    async def agenerate_batch(self, input_message: BaseMessage, output_parser: PydanticOutputParser, batch_size: int) -> list[PersistentSample]:
        
        coroutines = [self.client.ainvoke([input_message]) for m in range(batch_size)]    
        responses_batch: list[AIMessage] = await asyncio.gather(*coroutines)
        
        successfully_parsed = []
        for response in responses_batch:
            
            try:
                parsed, meta = output_parser.parse(response.content), response.usage_metadata
                successfully_parsed.append(PersistentSample(usage_metadata=meta, task_row_model=parsed))

            except OutputParserException as ex:
                logger.info("Error while parsing the LLM response. %s", ex)

        logger.info("Batch generated, batch_len=%s", len(successfully_parsed))
        return successfully_parsed

    
    async def apersist_sample(self, sample: PersistentSample, path: Path) -> None:

        path.mkdir(parents=True, exist_ok=True)

        uuid_path = path / f"{uuid.uuid4().hex}.json"

        async with aiofiles.open(uuid_path, mode="w", encoding="utf-8") as file:
            await file.write(sample.model_dump_json())
    

    def get_remaining_samples(self, combos: TCombo) -> TCombo:

        remaining = {}
        for key, samples_ready in combos.items():
            count_remaining = self.declaration.batch_size - samples_ready

            if count_remaining:
                remaining[key] = count_remaining 
        
        return remaining
