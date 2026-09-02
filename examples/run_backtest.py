"""Runs a strategy spec on the data already in the cache.

    python -m examples.run_backtest strategies/rsi-wick-baseline.json

No strategy parameter lives in here: everything comes from the JSON. The
command-line options only concern execution (initial equity, costs, time
window).
"""
from __future__ import annotations

import argparse
import logging
from datetime import timezone
from pathlib import Path

import pandas as pd

from core.data.cache import ParquetCache
from core.data.provider import SymbolSpec
from core.data.quality import check_quality
from core.engine.backtester import BacktestConfig, run_backtest
from core.engine.costs import CommissionModel, CostModel, SpreadPolicy, SwapModel
from core.metrics.performance import buy_and_hold, compute_metrics
from core.strategy.spec import StrategySpec

logger = logging.getLogger("run_backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="strategy JSON file")
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--years", type=int, nargs="*", help="years to load (default: all)")
    parser.add_argument("--start", type=str, default=None, help="ISO UTC")
    parser.add_argument("--end", type=str, default=None, help="ISO UTC")
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--commission", type=float, default=0.0,
                        help="per lot per side, account currency")
    parser.add_argument("--spread-mode", choices=("per_bar", "fixed", "quantile"),
                        default="per_bar")
    parser.add_argument("--spread-value", type=float, default=None)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--quality", action="store_true", help="also print the data report")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def load_bars(cache: ParquetCache, strategy: StrategySpec, years: list[int] | None) -> pd.DataFrame:
    symbol, timeframe = strategy.instrument.symbol, strategy.instrument.tf
    folder = cache.paths(symbol, timeframe, 0)[0].parent
    if years is None:
        years = sorted(int(p.stem) for p in folder.glob("*.parquet"))
    if not years:
        raise SystemExit(
            f"no cached data for {symbol} {timeframe.name} in {folder}. "
            f"Download it first with examples.download_year."
        )
    frames = [cache.read_year(symbol, timeframe, year) for year in years]
    return pd.concat([f for f in frames if len(f)]).sort_index()


def resolve_environment(
    strategy: StrategySpec, cache: ParquetCache, years: list[int] | None
) -> tuple[SymbolSpec, object]:
    """SymbolSpec and server timezone from the terminal, falling back to metadata."""
    from core.data.mt5_provider import MT5Provider

    symbol = strategy.instrument.symbol
    fallback_tz = None
    for year in years or []:
        meta = cache.read_meta(symbol, strategy.instrument.tf, year)
        if meta and meta.server_timezone:
            from zoneinfo import ZoneInfo

            fallback_tz = ZoneInfo(meta.server_timezone)
            break

    with MT5Provider(server_timezone=fallback_tz) as provider:
        spec = provider.get_symbol_spec(symbol)
        server_tz = provider.server_timezone
    return spec, server_tz


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    strategy = StrategySpec.from_json(args.spec)
    cache = ParquetCache(args.cache)

    folder = cache.paths(strategy.instrument.symbol, strategy.instrument.tf, 0)[0].parent
    years = args.years or sorted(int(p.stem) for p in folder.glob("*.parquet"))
    bars = load_bars(cache, strategy, years)
    if args.start:
        bars = bars[bars.index >= pd.Timestamp(args.start, tz=timezone.utc)]
    if args.end:
        bars = bars[bars.index < pd.Timestamp(args.end, tz=timezone.utc)]

    symbol_spec, server_tz = resolve_environment(strategy, cache, years)
    logger.info(
        "%s: point=%s tick_size=%s tick_value=%s contract=%s swap L/S=%s/%s",
        symbol_spec.name, symbol_spec.point, symbol_spec.tick_size,
        symbol_spec.tick_value, symbol_spec.contract_size,
        symbol_spec.swap_long, symbol_spec.swap_short,
    )

    if args.quality:
        report = check_quality(bars, symbol_spec.name, strategy.instrument.tf)
        for line in report.as_text().splitlines():
            logger.info("%s", line)

    costs = CostModel(
        spread=SpreadPolicy(mode=args.spread_mode, value=args.spread_value),
        commission=CommissionModel(args.commission),
        swap=SwapModel(mode="none" if args.no_swap else "points"),
    )
    config = BacktestConfig(initial_equity=args.equity, costs=costs)
    result = run_backtest(strategy, bars, symbol_spec, server_tz, config)

    strategy_report = compute_metrics(
        result.trades, result.equity, result.timeframe, args.equity, strategy.id
    )
    benchmark = buy_and_hold(
        bars, symbol_spec, strategy.sizing, result.timeframe, args.equity, costs, server_tz
    )

    for line in _render(strategy, result, strategy_report, benchmark).splitlines():
        logger.info("%s", line)


def _render(strategy, result, strategy_report, benchmark) -> str:
    lines = [
        "",
        "=" * 78,
        f"BACKTEST {strategy.id} - {strategy.instrument.symbol} "
        f"{strategy.instrument.timeframe}",
        "=" * 78,
        f"raw signals         : long {result.signals.counts['long']}, "
        f"short {result.signals.counts['short']}",
        f"executed trades     : {len(result.trades)}",
    ]
    if result.blocked:
        lines.append("blocked signals     :")
        for reason, count in result.blocked.most_common():
            lines.append(f"    {reason}: {count}")
    if len(result.trades):
        lines.append("exits by reason     :")
        for reason, count in result.trades["exit_reason"].value_counts().items():
            lines.append(f"    {reason}: {count}")
        lines.append(
            f"ambiguous trades (SL+TP within the same bar): {result.ambiguous_trades} "
            f"({result.ambiguous_trades / len(result.trades):.1%})"
        )
        lines.append(f"trades across a session gap             : {result.gap_crossing_trades}")
    lines.append("")
    lines.append(strategy_report.as_text())
    lines.append("")
    lines.append(benchmark.as_text())
    lines.append("=" * 78)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
