"""Decoder for the MT5 .hc cache (columnar format) -> OHLC DataFrame.

Useful when the MT5 terminal is not logged in and copy_rates_* is
unavailable: the on-disk cache still holds the history downloaded up to the
last connection.

Observed layout (build 5833, version 0x1F6):
  0x000  uint32 version
  0x004  copyright UTF-16
  ...
  0x1AC  consecutive columnar blocks, each: uint32 count + array
         [time int64][open f64][high f64][low f64][close f64]
         [tick_volume int64][spread int32][real_volume int64]

Timestamps are in server time, like copy_rates_*: `read_hc` therefore
returns a naive index, and `read_hc_utc` localizes it.
"""
from __future__ import annotations

import logging
import struct
from datetime import tzinfo
from pathlib import Path

import numpy as np
import pandas as pd

from core.data.provider import BAR_COLUMNS, normalize_bars
from core.data.servertime import server_epoch_to_utc_index

logger = logging.getLogger(__name__)

FIRST_BLOCK = 428


def _block(
    data: bytes, off: int, dtype: str, itemsize: int
) -> tuple[np.ndarray, int, int]:
    count: int = struct.unpack_from("<I", data, off)[0]
    start = off + 4
    arr = np.frombuffer(data, dtype=dtype, count=count, offset=start)
    return arr, start + count * itemsize, count


def read_hc(path: Path | str) -> pd.DataFrame:
    """Reads an .hc file and returns the bars with timestamps in server time."""
    data = Path(path).read_bytes()
    off = FIRST_BLOCK
    times, off, n = _block(data, off, "<i8", 8)
    cols: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close"):
        cols[name], off, c = _block(data, off, "<f8", 8)
        assert c == n, f"{name}: count {c} != {n}"
    cols["tick_volume"], off, c = _block(data, off, "<i8", 8)
    assert c == n
    cols["spread"], off, c = _block(data, off, "<i4", 4)
    assert c == n
    cols["real_volume"], off, c = _block(data, off, "<i8", 8)
    assert c == n

    df = pd.DataFrame({"time": pd.to_datetime(times, unit="s"), **cols})
    logger.info("read %s: %d bars (server time)", path, len(df))
    return df


def read_hc_utc(path: Path | str, server_tz: tzinfo) -> pd.DataFrame:
    """Like `read_hc`, but conforming to the DataProvider contract (UTC index)."""
    df = read_hc(path)
    frame = df.drop(columns="time")
    frame.index = server_epoch_to_utc_index(
        df["time"].astype("int64") // 10**9, server_tz
    )
    return normalize_bars(frame, BAR_COLUMNS)
