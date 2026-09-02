"""Performance metrics over trades and equity curve.

Two declared choices, because they change the numbers:

- the annualization factor is **derived from the session calendar** of the
  data (active weekly slots x 52.18), not a constant like 252 or 365;
- the risk-free rate is 0. On a leveraged intraday backtest that is the
  honest reference: what is being measured is the edge, not the alternative
  to a government bond.

Every strategy report must be read next to the buy-and-hold report over the
same period. A Sharpe of 0.4 means nothing until you know what the
instrument did by standing still.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

from core.data.provider import SymbolSpec, Timeframe
from core.data.quality import infer_session_slots
from core.engine.costs import CostModel, money_per_point
from core.engine.sizing import lots_for
from core.strategy.spec import Sizing

logger = logging.getLogger(__name__)

WEEKS_PER_YEAR = 365.25 / 7.0


def periods_per_year(
    index: pd.DatetimeIndex, timeframe: Timeframe, threshold: float = 0.5
) -> float:
    """Open-market bars in a year, derived from the data itself."""
    if len(index) < 2:
        return float(WEEKS_PER_YEAR * (7 * 24 * 60) / timeframe.minutes)
    active, _ = infer_session_slots(index, timeframe, threshold)
    slots_per_week = len(active) or (7 * 24 * 60 // timeframe.minutes)
    return float(slots_per_week * WEEKS_PER_YEAR)


@dataclass
class PerformanceReport:
    """All the metrics of a single result series."""

    label: str
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    days: float
    initial_equity: float
    final_equity: float
    total_return: float
    annual_return: float
    sharpe: float
    sortino: float
    max_drawdown_money: float
    max_drawdown_pct: float
    calmar: float
    trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_duration: timedelta | None
    exposure: float
    max_losing_streak: int
    t_stat: float
    p_value: float
    r_multiples: dict[str, float] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)
    ambiguous_trades: int = 0
    gap_crossing_trades: int = 0

    def as_text(self) -> str:
        duration = (
            f"{self.avg_duration}" if self.avg_duration is not None else "n/a"
        )
        lines = [
            f"[{self.label}]",
            f"  period              : {self.start} -> {self.end} ({self.days:.0f} days)",
            f"  equity              : {self.initial_equity:.2f} -> {self.final_equity:.2f} "
            f"({self.total_return:+.2%})",
            f"  annual return       : {self.annual_return:+.2%}",
            f"  Sharpe / Sortino    : {self.sharpe:.2f} / {self.sortino:.2f}",
            f"  max drawdown        : {self.max_drawdown_money:.2f} "
            f"({self.max_drawdown_pct:.2%})",
            f"  Calmar              : {self.calmar:.2f}",
            f"  trades              : {self.trades}",
            f"  win rate            : {self.win_rate:.2%}",
            f"  profit factor       : {self.profit_factor:.3f}",
            f"  expectancy / trade  : {self.expectancy:+.4f}",
            f"  avg duration        : {duration}",
            f"  exposure            : {self.exposure:.2%}",
            f"  max losing streak   : {self.max_losing_streak}",
            f"  t-stat (p-value)    : {self.t_stat:.2f} ({self.p_value:.4f})",
        ]
        if self.r_multiples:
            lines.append(
                f"  R-multiples         : mean {self.r_multiples['mean']:+.3f}, "
                f"median {self.r_multiples['median']:+.3f}, "
                f"p05 {self.r_multiples['p05']:+.3f}, p95 {self.r_multiples['p95']:+.3f}"
            )
        if self.costs:
            lines.append(
                f"  costs               : spread {self.costs['spread']:.2f}, "
                f"commission {self.costs['commission']:.2f}, "
                f"swap {self.costs['swap']:.2f} "
                f"(gross {self.costs['gross']:+.2f} -> net {self.costs['net']:+.2f})"
            )
        if self.max_drawdown_pct > 1.0 or self.final_equity <= 0:
            lines.append(
                "  WARNING             : drawdown beyond 100%, the equity went "
                "below zero. With this sizing a margin call would have arrived: "
                "the metrics above are arithmetic, not an achievable result."
            )
        if self.ambiguous_trades or self.gap_crossing_trades:
            lines.append(
                f"  ambiguous trades    : {self.ambiguous_trades} "
                f"(SL and TP touched within the same bar)"
            )
            lines.append(f"  gap-crossing trades : {self.gap_crossing_trades}")
        return "\n".join(lines)


def _drawdown(equity: pd.Series) -> tuple[float, float]:
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    drawdown = equity - peak
    worst = float(drawdown.min())
    relative = drawdown / peak.replace(0.0, np.nan)
    return abs(worst), abs(float(relative.min())) if relative.notna().any() else 0.0


def _risk_adjusted(equity: pd.Series, ppy: float) -> tuple[float, float]:
    returns = equity.pct_change().dropna()
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0, 0.0
    scale = float(np.sqrt(ppy))
    sharpe = float(returns.mean() / returns.std(ddof=1) * scale)
    downside = returns[returns < 0]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return sharpe, 0.0
    sortino = float(returns.mean() / downside.std(ddof=1) * scale)
    return sharpe, sortino


def _max_losing_streak(pnl: pd.Series) -> int:
    longest = current = 0
    for value in pnl:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def compute_metrics(
    trades: pd.DataFrame,
    equity: pd.Series,
    timeframe: Timeframe,
    initial_equity: float,
    label: str = "strategy",
    exposure_bars: int | None = None,
) -> PerformanceReport:
    """Full report from trades + equity curve."""
    start = equity.index[0] if len(equity) else None
    end = equity.index[-1] if len(equity) else None
    days = (end - start).total_seconds() / 86400.0 if start is not None else 0.0
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    total_return = final / initial_equity - 1.0 if initial_equity else 0.0

    years = days / 365.25
    if years > 0 and final > 0 and initial_equity > 0:
        annual_return = (final / initial_equity) ** (1.0 / years) - 1.0
    else:
        annual_return = 0.0

    ppy = periods_per_year(pd.DatetimeIndex(equity.index), timeframe) if len(equity) else 1.0
    sharpe, sortino = _risk_adjusted(equity, ppy)
    dd_money, dd_pct = _drawdown(equity)
    calmar = annual_return / dd_pct if dd_pct > 0 else 0.0

    count = int(len(trades))
    if count:
        pnl = trades["net_pnl"].astype("float64")
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = len(wins) / count
        profit_factor = (
            float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0
            else float("inf") if len(wins) else 0.0
        )
        expectancy = float(pnl.mean())
        duration = (trades["exit_time"] - trades["entry_time"]).mean()
        avg_duration = duration.to_pytimedelta() if pd.notna(duration) else None
        held = int(trades["bars_held"].sum())
        exposure = held / (exposure_bars or len(equity) or 1)
        streak = _max_losing_streak(pnl)
        if len(pnl) > 1 and pnl.std(ddof=1) > 0:
            t_stat = float(pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))))
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(pnl) - 1)))
        else:
            t_stat = p_value = float("nan")
        r = trades["r_multiple"].astype("float64").dropna()
        r_multiples = (
            {
                "mean": float(r.mean()),
                "median": float(r.median()),
                "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
                "p05": float(r.quantile(0.05)),
                "p95": float(r.quantile(0.95)),
                "min": float(r.min()),
                "max": float(r.max()),
            }
            if len(r)
            else {}
        )
        costs = {
            "spread": float(trades["spread_cost"].sum()),
            "commission": float(trades["commission"].sum()),
            "swap": float(trades["swap"].sum()),
            "gross": float(trades["gross_pnl"].sum()),
            "net": float(pnl.sum()),
        }
        ambiguous = int(trades["ambiguous"].sum())
        gap_crossing = int(trades["crossed_gap"].sum())
    else:
        win_rate = profit_factor = expectancy = exposure = 0.0
        avg_duration = None
        streak = 0
        t_stat = p_value = float("nan")
        r_multiples = {}
        costs = {}
        ambiguous = gap_crossing = 0

    return PerformanceReport(
        label=label,
        start=start,
        end=end,
        days=days,
        initial_equity=initial_equity,
        final_equity=final,
        total_return=total_return,
        annual_return=annual_return,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown_money=dd_money,
        max_drawdown_pct=dd_pct,
        calmar=calmar,
        trades=count,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_duration=avg_duration,
        exposure=exposure,
        max_losing_streak=streak,
        t_stat=t_stat,
        p_value=p_value,
        r_multiples=r_multiples,
        costs=costs,
        ambiguous_trades=ambiguous,
        gap_crossing_trades=gap_crossing,
    )


def buy_and_hold(
    bars: pd.DataFrame,
    symbol_spec: SymbolSpec,
    sizing: Sizing,
    timeframe: Timeframe,
    initial_equity: float,
    costs: CostModel,
    server_tz,
) -> PerformanceReport:
    """Buy on the first bar and stand still until the last.

    Same costs as the strategy: spread paid on entry, round-turn commission,
    swap for every night of the period. This is the comparison that says
    whether the strategy is adding anything or just paying spread.
    """
    if bars.empty:
        return compute_metrics(
            pd.DataFrame(), pd.Series(dtype="float64"), timeframe, initial_equity, "buy & hold"
        )

    lots = lots_for(sizing, initial_equity, symbol_spec)
    if lots <= 0:
        logger.warning("buy & hold: equity below the minimum lot")
        lots = symbol_spec.volume_min

    value = money_per_point(symbol_spec, lots)
    spread_points = costs.spread.series(bars)
    entry_raw = float(bars["open"].iloc[0])
    entry_spread = float(spread_points.iloc[0])
    commission = costs.commission.round_turn(lots)

    gross = (bars["close"] - entry_raw) / symbol_spec.point * value
    swap = costs.swap.charge(
        symbol_spec, 1, lots, bars.index[0].to_pydatetime(),
        bars.index[-1].to_pydatetime(), server_tz,
    )
    # swap accrues over time: spreading it linearly avoids showing a clean
    # curve that suddenly collapses on the last bar
    accrual = np.linspace(0.0, swap, len(bars))
    equity = initial_equity + gross - entry_spread * value - commission + accrual
    equity.name = "equity"

    exit_raw = float(bars["close"].iloc[-1])
    gross_total = (exit_raw - entry_raw) / symbol_spec.point * value
    trade = pd.DataFrame(
        [
            {
                "direction": 1,
                "entry_time": bars.index[0],
                "entry_price": entry_raw + entry_spread * symbol_spec.point,
                "exit_time": bars.index[-1],
                "exit_price": exit_raw,
                "exit_reason": "end_of_data",
                "lots": lots,
                "bars_held": len(bars),
                "session_bars_held": len(bars),
                "gross_pnl": gross_total,
                "spread_points": entry_spread,
                "spread_cost": entry_spread * value,
                "commission": commission,
                "swap": swap,
                "net_pnl": gross_total - entry_spread * value - commission + swap,
                "risk_money": np.nan,
                "r_multiple": np.nan,
                "ambiguous": False,
                "crossed_gap": False,
            }
        ]
    )
    return compute_metrics(
        trade, equity, timeframe, initial_equity, "buy & hold", exposure_bars=len(bars)
    )
