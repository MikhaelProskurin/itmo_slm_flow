"""Synthetic dataset generation using LLM APIs.

Provides a configurable, async generator that iterates over task, domain, and
difficulty combinations to produce structured training examples via
``langchain``-based prompt chains with automatic parsing and fallback logic.
"""

import os
import uuid
import asyncio
import aiofiles
import logging

from typing import Sequence

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from core.io.models import SyntheticRowDump


logger = logging.getLogger(__name__)


class MessagesBuilder:
    
    def __init__(self, templates: dict[str, str], parsers: dict[str, PydanticOutputParser]) -> None:
        self.templates = templates
        self.parsers = parsers

    @classmethod
    def from_sequence(cls, *args: tuple[str, str, BaseModel]) -> "MessagesBuilder":

        templates = {}
        parsers = {}
        for sequence in args:
            key, template, output_model = sequence
            templates[key] = template
            parsers[key] = PydanticOutputParser(pydantic_object=output_model)

        return cls(templates, parsers)
    
    def create_message(self, task: str, domain: str, difficulty: str) -> BaseMessage:
        template, parser = self.templates[task], self.parsers[task]
        fmt = parser.get_format_instructions()
        return SystemMessage(
            template.format(
            domain=domain, 
            difficulty=difficulty, 
            fmt=fmt
            )
        )

    def get_parser(self, key: str) -> PydanticOutputParser:
        return self.parsers[key]
        

class DatasetDeclaration(BaseModel):
    
    tasks: Sequence[str]
    domains: Sequence[str]
    difficulties: Sequence[str]
    batch_size: int = 10

    @property
    def n_samples(self) -> int:
        """Total number of examples that will be generated across all combinations."""
        return len(self.tasks) * len(self.domains) * len(self.difficulties) * self.batch_size
  

class AsyncDeclarativeDatasetGenerator:
    
    def __init__(self, client: ChatOpenAI, declaration: DatasetDeclaration, messages_builder: MessagesBuilder, rate_limit: int = 15) -> None:
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
        messages = [self.messages_builder.create_message(*c) for c in combos]
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
                
                path = output_dir + "/" + "/".join(key)
                persistence_coroutines = [self.apersist_sample(sample, path) for sample in batch]
                await asyncio.gather(*persistence_coroutines)
            
                results.extend(batch)
            
        return results


    async def agenerate_batch(self, input_message: BaseMessage, output_parser: PydanticOutputParser, batch_size: int) -> list[SyntheticRowDump]:
        
        coroutines = [self.client.ainvoke([input_message]) for m in range(batch_size)]    
        responses_batch: list[AIMessage] = await asyncio.gather(*coroutines)
        
        successfully_parsed = []
        for response in responses_batch:
            
            try:
                parsed, meta = output_parser.parse(response.content), response.usage_metadata
                successfully_parsed.append(SyntheticRowDump(usage_metadata=meta, task_row_model=parsed))

            except OutputParserException as ex:
                logger.info("Error while parsing the LLM response. %s", ex)

        logger.info("Batch generated, batch_len=%s", len(successfully_parsed))
        return successfully_parsed

    
    async def apersist_sample(self, sample: SyntheticRowDump, path: str) -> None:

        if not os.path.exists(path):
            os.makedirs(path, mode=777)
        
        uuid_path = path + "/" + uuid.uuid4().hex + ".json"
        async with aiofiles.open(uuid_path, mode="w", encoding="utf-8") as file:
            await file.write(sample.model_dump_json())
    

    def get_remaining_samples(self, combos: dict[tuple, int]) -> dict[tuple, int]:

        remaining = {}
        for key, samples_ready in combos.items():
            count_remaining = self.declaration.batch_size - samples_ready

            if count_remaining:
                remaining[key] = count_remaining 
        
        return remaining
