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

from typing import Sequence, Dict

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

from core.io.models import SyntheticRowDump


logger = logging.getLogger(__name__)


class DatasetGenerationConfig(BaseModel):
    """Configuration schema for the synthetic dataset generation run.

    Defines the Cartesian product of tasks, domains, and difficulties that
    the generator iterates over, along with per-combination batch size.

    param: tasks: RAG task types to generate examples for (e.g. ``["reranking"]``).
       type: Sequence[str]
    param: domains: Knowledge domains for the generated queries (e.g. ``["math", "medicine"]``).
       type: Sequence[str]
    param: difficulties: Difficulty levels applied to each combination (e.g. ``["easy", "hard"]``).
       type: Sequence[str]
    param: batch_size: Number of examples generated per (task, domain, difficulty) combination.
       type: int
    """

    tasks: Sequence[str]
    domains: Sequence[str]
    difficulties: Sequence[str]
    batch_size: int = 15

    @property
    def n_samples(self) -> int:
        """Total number of examples that will be generated across all combinations."""
        return len(self.tasks) * len(self.domains) * len(self.difficulties) * self.batch_size


class SyntheticDataGenerator:
    """Asynchronously generates synthetic training examples via an LLM API.

    Schedules all (task, domain, difficulty) batches concurrently, enforcing
    a semaphore-based concurrency limit to avoid overwhelming the API.

    param: llm_api_client: LangChain-compatible OpenAI client used for generation.
       type: ChatOpenAI
    param: config: Generation configuration defining tasks, domains, difficulties, and batch size.
       type: DatasetGenerationConfig
    param: parsers: Per-task Pydantic output parsers keyed by task name.
       type: Dict[str, PydanticOutputParser]
    param: prompts: Registry of prompt templates; keys must match ``parsers``.
       type: PromptRegistry
    param: max_concurrency: Maximum number of simultaneous LLM API calls.
       type: int
    param: max_retries: Number of repair-prompt retries on OutputParserException before discarding.
       type: int
    """

    def __init__(self,
        llm_api_client: ChatOpenAI,
        config: DatasetGenerationConfig,
        parsers: Dict[str, PydanticOutputParser],
        prompts: Dict[str, str],
        max_concurrency: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.llm = llm_api_client
        self.config = config

        if parsers.keys() != prompts.keys():
            raise ValueError("Parsers and prompts must have the same keys")

        self.parsers = parsers
        self.prompts = prompts

        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.generated_cnt: int = 0

    async def generate_synthetic_dataset(self, output_dir: str = "./tmp") -> list[SyntheticRowDump]:
        """Run all batches concurrently and return the full list of generated rows.

        param: output_dir: Root directory where per-combination JSON files are written.
           type: str
        """
        combos = [
            (task, domain, difficulty)
            for task in self.config.tasks
            for domain in self.config.domains
            for difficulty in self.config.difficulties
        ]
        coroutines = [self.generate_batch(*c, self.config.batch_size, output_dir) for c in combos]
        rows = sum(await asyncio.gather(*coroutines), [])

        return rows

    async def generate_batch(
        self,
        task: str,
        domain: str,
        difficulty: str,
        batch_size: int,
        output_dir: str
    ) -> list[SyntheticRowDump]:
        """Generate and persist one batch of examples for a single (task, domain, difficulty) combo.

        param: task: Task name used to look up the parser and prompt template.
           type: str
        param: domain: Knowledge domain injected into the prompt.
           type: str
        param: difficulty: Difficulty level injected into the prompt.
           type: str
        param: batch_size: Number of examples to generate for this combination.
           type: int
        param: output_dir: Root output directory; files are saved under ``output_dir/task/domain/difficulty/``.
           type: str
        """
        parser = self.parsers[task]
        template = self.prompts[task]

        fmt_kwargs = {"domain": domain, "difficulty": difficulty, "fmt": parser.get_format_instructions()}
        json_output_path = "/".join([output_dir, task, domain, difficulty])

        prompt = [SystemMessage(template.format(**fmt_kwargs))]

        logger.info(f"Scheduling {batch_size} coroutines for [{task}, {domain}, {difficulty}]")

        coroutines = [self.save_with_fallback(prompt, parser, json_output_path, self.max_retries) for _ in range(batch_size)]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        batch: list[SyntheticRowDump] = []
        failed = 0
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Item failed in [{task}, {domain}, {difficulty}]: {result}")
                failed += 1
            else:
                batch.append(result)

        self.generated_cnt += len(batch)
        logger.info(
            f"Batch done [{task}, {domain}, {difficulty}]: "
            f"{len(batch)} ok, {failed} failed "
            f"[{self.generated_cnt}/{self.config.n_samples}]"
        )

        return batch

    async def save_with_fallback(
        self,
        prompt: list[BaseMessage],
        parser: PydanticOutputParser,
        json_output_path: str = "./tmp",
        max_retries: int = 3,
    ) -> SyntheticRowDump:
        """Invoke the LLM, parse the response, and persist the result to a UUID-named JSON file.

        On ``OutputParserException``, appends the malformed output and a correction
        request to the conversation and retries up to ``max_retries`` times. Raises
        ``OutputParserException`` only after all attempts are exhausted so the caller
        can count and log failures without crashing the batch.

        param: prompt: Formatted message list sent to the LLM.
           type: list[BaseMessage]
        param: parser: Pydantic parser used to validate and deserialize the LLM output.
           type: PydanticOutputParser
        param: json_output_path: Directory path where the JSON checkpoint is written.
           type: str
        param: max_retries: Number of repair-prompt retries before discarding the example.
           type: int
        """
        messages: list[BaseMessage] = list(prompt)
        content = None
        model_response: AIMessage | None = None

        for attempt in range(max_retries + 1):
            async with self.semaphore:
                model_response = await self.llm.ainvoke(messages)

            try:
                content = parser.parse(model_response.content)
                break
            except OutputParserException as ex:
                if attempt < max_retries:
                    messages = messages + [
                        AIMessage(content=model_response.content),
                        HumanMessage(
                            content=(
                                "Your previous response could not be parsed. "
                                "Fix it and respond using the exact format below:\n"
                                f"{parser.get_format_instructions()}"
                            )
                        ),
                    ]
                else:
                    logger.warning(f"All {max_retries + 1} attempts failed, discarding example: {ex}")
                    raise

        serialized = SyntheticRowDump(
            usage_metadata=model_response.usage_metadata,
            response_metadata=model_response.response_metadata,
            task_row_model=content
        )

        if not os.path.exists(json_output_path):
            os.makedirs(json_output_path, 777)

        uuid_path = json_output_path + "/" + uuid.uuid4().hex + ".json"
        async with aiofiles.open(uuid_path, mode="w", encoding="utf-8") as f:
            await f.write(serialized.model_dump_json())

        return serialized
