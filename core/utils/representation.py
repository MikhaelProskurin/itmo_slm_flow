"""Utilities for formatting and displaying pipeline results as Pandas DataFrames."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from pydantic import BaseModel

def plot_curve_by_artifacts(
        x: list[float],
        y: list[float]
) -> None:
    plt.plot()
    return

def dump_to_csv(data: list[type[BaseModel]], path: str, create_temp_dir: bool = True) -> None:
    """Serialize a list of Pydantic models to a CSV file.

    Args:
        data: Records to serialize; each item is dumped via ``model_dump()``.
        path: Destination path without extension; ``.csv`` is appended automatically.
        create_temp_dir:
    """
    _tmp_location = Path("./_tmp")

    if create_temp_dir:
        _tmp_location.mkdir(parents=True, exist_ok=True)
        path = str(_tmp_location / path)

    dataframe = pd.DataFrame([row.model_dump() for row in data])
    dataframe.to_csv(path + ".csv", index=False)
    return