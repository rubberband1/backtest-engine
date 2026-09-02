"""Conversion of Python values into something JSON actually accepts.

`inf` and `NaN` are legitimate, informative floats (an infinite profit factor
means "no losing trades", a NaN means "not computable"), but they are not
valid JSON: the browser's `JSON.parse` rejects them. They become `null`, and
the reader tells the cases apart by looking at the other metrics.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Makes a value serializable, recursively."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [json_safe(v) for v in value]
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    return value
