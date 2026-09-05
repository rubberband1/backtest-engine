"""Pointing a spec at an instrument, in one place.

The instrument block of a `StrategySpec` is not decoration. The engine reads
the timeframe from it to count session bars for the time stop, the spread
realism check is run for it, and the result is labelled with it - so a spec
authored on `XAUUSD.r H1` and executed over `AUDUSD.r H4` bars runs as an H1
strategy on H4 data, with a time stop out by the ratio of the two and nothing
in the output saying so.

That was a real defect, and it existed because there was more than one way to
bind: the campaign runner went through `bind_cell`, while the API, the batch
runner and the live runner each wrote the instrument block by hand. Only the
campaign path was right.

There is one binder now. `BoundSpec` rebuilds the instrument block from the
symbol and timeframe it is constructed with, so a bound spec cannot disagree
with the cell it names, and every entry point that executes a spec - backtest,
edge, preview, screen, batch, live, replay - takes one of these instead of a
bare `StrategySpec`. A path that skips the binder does not type-check.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from core.data.provider import Timeframe
from core.strategy.spec import StrategySpec


class CellMismatch(ValueError):
    """A bound spec was handed to a run over other bars than the ones it names."""


@dataclass(frozen=True)
class BoundSpec:
    """A spec and the cell it is executed on, which cannot disagree.

    Construction is the binding: the instrument block is rewritten from
    `symbol` and `timeframe`, so there is no way to hold one of these whose
    spec points elsewhere.
    """

    spec: StrategySpec
    symbol: str
    timeframe: str

    def __post_init__(self) -> None:
        timeframe = Timeframe.parse(self.timeframe).name
        object.__setattr__(self, "timeframe", timeframe)
        instrument = self.spec.instrument
        if instrument.symbol == self.symbol and instrument.timeframe == timeframe:
            return
        payload = copy.deepcopy(self.spec.model_dump(mode="json"))
        payload["instrument"]["symbol"] = self.symbol
        payload["instrument"]["timeframe"] = timeframe
        object.__setattr__(self, "spec", StrategySpec.from_dict(payload, "<bound>"))

    @property
    def tf(self) -> Timeframe:
        return Timeframe.parse(self.timeframe)

    def must_match(self, symbol: str, timeframe: str | None = None) -> None:
        """Raises unless this is bound to exactly the bars about to be run.

        `timeframe` is omitted where the caller has no declared timeframe to
        check against - a DataFrame does not carry one, so the live runner and
        the replay can only cross-check the instrument spec they were handed.
        """
        wanted = None if timeframe is None else Timeframe.parse(timeframe).name
        if self.symbol == symbol and (wanted is None or self.timeframe == wanted):
            return
        raise CellMismatch(
            f"the spec is bound to {self.symbol} {self.timeframe}, but the run "
            f"is over {symbol} {wanted or self.timeframe}: bind it to the cell "
            f"it is executed on"
        )


def bind_cell(spec: StrategySpec, symbol: str, timeframe: str) -> BoundSpec:
    """The spec as it is actually run on this cell.

    The symbol is bound too, so the run folder says which instrument it was.
    """
    return BoundSpec(spec, symbol, timeframe)
