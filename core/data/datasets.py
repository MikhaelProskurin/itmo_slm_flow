"""Dataset wrappers for loading and accessing synthetic slm-flow examples.

Provides an abstract base and a concrete file-based implementation that
recursively loads JSON files from the dataset directory into typed Pydantic rows.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from abc import ABC, abstractmethod

from pandas import DataFrame

from core.io.models import SlmFlowDatasetRow


class SlmFlowBaseDataset(ABC):
    """Abstract interface for slm-flow dataset wrappers."""

    @classmethod
    @abstractmethod
    def from_files(cls, directory: str, v1_compatible: bool) -> SlmFlowBaseDataset:
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
    def __getitem__(self, idx: int) -> SlmFlowDatasetRow:
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class SlmFlowSyntheticDataset(SlmFlowBaseDataset):
    """File-based dataset that loads JSON examples from the ``slm_flow_df/`` directory tree.

    param: rows: Pre-loaded list of dataset rows.
       type: list[SlmFlowDatasetRow]
    param: v1_compatible: When ``True``, treats the raw JSON object as ``task_row_model`` (legacy format).
       type: bool
    """

    def __init__(self, rows: list[SlmFlowDatasetRow], v1_compatible: bool) -> None:
        self.rows = rows
        self.v1_compatible = v1_compatible
        random.shuffle(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> SlmFlowDatasetRow:
        return self.rows[idx]

    @classmethod
    def from_files(cls, directory: str, v1_compatible: bool = True) -> SlmFlowSyntheticDataset:
        """Recursively load all ``*.json`` files under ``directory`` into typed rows.

        param: directory: Root path of the dataset (e.g. ``"slm_flow_df"``).
           type: str
        param: v1_compatible: Forward to ``json_to_pydantic`` for legacy JSON layout support.
           type: bool
        """
        directory_path_object = Path(directory)

        rows: list[SlmFlowDatasetRow] = []
        for filename in directory_path_object.rglob("*.json"):
            rows.append(cls.json_to_pydantic(filename, v1_compatible))

        return cls(rows, v1_compatible)

    @property
    def to_pandas(self) -> DataFrame:
        """Serialize all rows to a Pandas DataFrame via ``model_dump``."""
        return DataFrame([row.model_dump() for row in self.rows])

    @classmethod
    def json_to_pydantic(cls, filename: str | Path, v1_compatible: bool) -> SlmFlowDatasetRow:
        """Parse a single JSON file into a ``SlmFlowDatasetRow``, inferring metadata from the path.

        param: filename: Absolute or relative path to the JSON file; must follow the
           ``root/task/domain/difficulty/uuid.json`` directory convention.
           type: str | Path
        param: v1_compatible: When ``True``, uses the full JSON object as ``task_row_model``;
           otherwise reads the nested ``task_row_model`` key.
           type: bool
        """
        _, root, task, domain, difficulty, uuid = filename.parts
        json_location = "/".join(filename.parts)

        with open(json_location, mode="r", encoding="utf-8") as f:
            content = json.load(f)

            return SlmFlowDatasetRow(
                task=task,
                domain=domain,
                difficulty=difficulty,
                usage_metadata=content.get("usage_metadata"),
                response_metadata=content.get("response_metadata"),
                task_row_model=content if v1_compatible else content.get("task_row_model")
            )
