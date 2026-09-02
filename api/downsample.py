"""Point reduction of a series before sending it to the browser.

A one-year M1 equity curve has 265,000 points: sending them all means 20 MB
of JSON to draw 900 pixels. The cut must happen **after** computing what is
needed (the drawdown is computed on the whole series, not the reduced one),
and it must preserve the peaks: a naive decimation hides exactly the highs
and lows, i.e. the only part of the curve anyone cares about.

LTTB (Largest Triangle Three Buckets) picks, in each interval, the point
forming the largest-area triangle with the previously chosen point and the
average of the next bucket. It preserves shape and vertices.
"""
from __future__ import annotations

import numpy as np


def lttb_indices(x: np.ndarray, y: np.ndarray, target: int) -> np.ndarray:
    """Indices of the points to keep, first and last always included."""
    n = len(x)
    if target >= n or target < 3:
        return np.arange(n)

    xs = x.astype("float64")
    ys = np.nan_to_num(y.astype("float64"), nan=0.0)

    kept = np.empty(target, dtype="int64")
    kept[0] = 0
    kept[-1] = n - 1
    # buckets cover the interior points; the first and last are already fixed
    bucket_size = (n - 2) / (target - 2)
    previous = 0

    for i in range(target - 2):
        start = int(np.floor(i * bucket_size)) + 1
        end = int(np.floor((i + 1) * bucket_size)) + 1
        end = min(end, n - 1)
        if start >= end:
            kept[i + 1] = min(start, n - 2)
            previous = kept[i + 1]
            continue

        next_start = end
        next_end = min(int(np.floor((i + 2) * bucket_size)) + 1, n)
        if next_start >= next_end:
            next_start, next_end = n - 1, n
        avg_x = xs[next_start:next_end].mean()
        avg_y = ys[next_start:next_end].mean()

        px, py = xs[previous], ys[previous]
        areas = np.abs(
            (px - avg_x) * (ys[start:end] - py) - (px - xs[start:end]) * (avg_y - py)
        )
        chosen = start + int(np.argmax(areas))
        kept[i + 1] = chosen
        previous = chosen

    return np.unique(kept)


def downsample(x: np.ndarray, y: np.ndarray, target: int) -> np.ndarray:
    """Indices to keep to represent the series with ~`target` points."""
    if len(x) == 0:
        return np.zeros(0, dtype="int64")
    return lttb_indices(x, y, target)
