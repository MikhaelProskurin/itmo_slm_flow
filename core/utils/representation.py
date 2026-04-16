"""Utilities for formatting and displaying pipeline results as Pandas DataFrames."""

import pandas as pd
import matplotlib.pyplot as plt

from pydantic import BaseModel

def plot_curve_by_artifacts(x, y) -> None:
    fig, ax = ...
    plt.show()

def dump_to_csv(data: list[BaseModel], path: str) -> None:
    """Serialize a list of Pydantic models to a CSV file.

    Args:
        data: Records to serialize; each item is dumped via ``model_dump()``.
        path: Destination path without extension; ``.csv`` is appended automatically.
    """
    dataframe = pd.DataFrame([row.model_dump() for row in data])
    dataframe.to_csv(path + ".csv", index=False)
    return