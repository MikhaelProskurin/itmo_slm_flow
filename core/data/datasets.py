"""Dataset wrappers for loading and accessing synthetic slm-flow examples.

Provides an abstract base and a concrete file-based implementation that
recursively loads JSON files from the dataset directory into typed Pydantic rows.
"""

import json
import random

from typing import Any
from pathlib import Path

from pandas import DataFrame
from pydantic import BaseModel

from abc import ABC, abstractmethod
from core.data import RAGDocument

TPythonMap = dict[str, Any]

class StandardSample(BaseModel):
    """Canonical in-memory representation of a single RAG example (query, documents, answer)."""

    query: str
    documents: list[RAGDocument]
    golden_answer: str


class DatasetRecord(BaseModel):
    """Unified dataset row loaded from disk and fed into the inference pipeline."""

    task: str
    domain: str
    difficulty: str
    usage_metadata: TPythonMap | None
    sample: StandardSample


class BaseDataset(ABC):
    """Abstract interface for slm-flow dataset wrappers."""

    @classmethod
    @abstractmethod
    def from_files(cls, directory: str) -> "BaseDataset":
        """Construct the dataset by loading all JSON files from ``directory``."""
        pass

    @property
    @abstractmethod
    def to_pandas(self) -> DataFrame:
        """Return the dataset as a Pandas DataFrame."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> DatasetRecord:
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class RAGSyntheticDataset(BaseDataset):
    """File-backed dataset that loads all JSON examples from a directory tree and shuffles them.

    Expected directory structure: ``{root}/{task}/{domain}/{difficulty}/{uuid}.json``.
    Metadata (task, domain, difficulty) is inferred from the path components.
    """

    def __init__(self, rows: list[DatasetRecord]) -> None:
        self.rows = rows
        random.shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatasetRecord:
        return self.rows[idx]

    @classmethod
    def from_files(cls, directory: str | Path) -> "RAGSyntheticDataset":
        """Recursively load all ``*.json`` files under ``directory`` and return a dataset instance."""
        rows = [cls.json_to_pydantic(file) for file in Path(directory).rglob("*.json")]
        return cls(rows)

    @property
    def to_pandas(self) -> DataFrame:
        """Return all rows as a Pandas DataFrame with one row per dataset record."""
        return DataFrame([row.model_dump() for row in self.rows])

    @staticmethod
    def json_to_pydantic(filename: Path) -> DatasetRecord:
        """Parse a single JSON file into a ``DatasetRecord``, inferring metadata from its path."""
        _, root, task, domain, difficulty, uuid = filename.parts

        with filename.open(mode="r", encoding="utf-8") as f:
            content = json.load(f)

            return DatasetRecord(
                task=task,
                domain=domain,
                difficulty=difficulty,
                usage_metadata=content.get("usage_metadata"),
                sample=StandardSample.model_validate(content.get("sample"))
            )
