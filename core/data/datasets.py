"""Dataset wrappers for loading and accessing synthetic slm-flow examples.

Provides an abstract base and a concrete file-based implementation that
recursively loads JSON files from the dataset directory into typed Pydantic rows.
"""

import json
import random
from typing import Any

from abc import ABC, abstractmethod

from pathlib import Path
from pydantic import BaseModel

from pandas import DataFrame

TContentMap = dict[str, Any]

class DatasetRecord(BaseModel):
    """Unified dataset row loaded from disk and fed into the inference pipeline."""

    task: str
    domain: str
    difficulty: str
    usage_metadata: TContentMap | None
    sample: TContentMap


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

    def __init__(self, rows: list[DatasetRecord]) -> None:
        self.rows = rows
        random.shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> DatasetRecord:
        return self.rows[idx]

    @classmethod
    def from_files(cls, directory: str | Path) -> "RAGSyntheticDataset":
        rows = [cls.json_to_pydantic(file) for file in Path(directory).rglob("*.json")]
        return cls(rows)

    @property
    def to_pandas(self) -> DataFrame:
        return DataFrame([row.model_dump() for row in self.rows])

    @classmethod
    def json_to_pydantic(cls, filename: Path) -> DatasetRecord:

        _, root, task, domain, difficulty, uuid = filename.parts

        with filename.open(mode="r", encoding="utf-8") as f:
            content = json.load(f)

            return DatasetRecord(
                task=task,
                domain=domain,
                difficulty=difficulty,
                usage_metadata=content.get("usage_metadata"),
                sample=content.get("sample")
            )
