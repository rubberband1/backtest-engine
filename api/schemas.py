"""Request and response models of the API.

They are explicit and complete because the OpenAPI schema is born here, and
from it the frontend's TypeScript types: a `dict[str, Any]` here becomes an
`unknown` over there, and the frontend loses all checking.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SpreadMode = Literal["per_bar", "fixed", "quantile"]
SwapMode = Literal["points", "money", "none"]
RunStatus = Literal["running", "done", "error"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- instruments ---------------------------------------------------------


class SymbolSpecOut(Model):
    name: str
    point: float
    digits: int
    contract_size: float
    tick_value: float
    tick_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    swap_long: float
    swap_short: float
    currency_profit: str
    trade_mode: str


class SymbolListOut(Model):
    symbols: list[SymbolSpecOut]
    source: Literal["terminal", "cache"]
    server_timezone: str | None = None
    cached_symbols: list[str] = Field(
        default_factory=list, description="symbols with data already in the local cache"
    )


class GapOut(Model):
    start: datetime
    end: datetime
    missing_bars: int


class QualityOut(Model):
    rows: int
    first_bar: datetime | None
    last_bar: datetime | None
    expected_bars: int
    missing_bars: int
    completeness: float
    gaps: int
    duplicate_timestamps: int
    zero_spread: int
    session_confidence: str
    text: str
    worst_gaps: list[GapOut] = Field(default_factory=list)


class CoverageOut(Model):
    symbol: str
    timeframe: str
    years: list[int]
    start: datetime | None
    end: datetime | None
    bars: int
    quality: QualityOut | None = None


# -- strategies ----------------------------------------------------------


class StrategyOut(Model):
    id: str
    name: str
    description: str
    symbol: str
    timeframe: str
    file: str
    spec: dict[str, Any]


class ValidateRequest(Model):
    spec: dict[str, Any]


class ValidateResponse(Model):
    valid: bool
    message: str
    errors: list[str] = Field(default_factory=list)
    normalized: dict[str, Any] | None = None


# -- run configuration ---------------------------------------------------


class RunConfigIn(Model):
    symbol: str = Field(min_length=1)
    timeframe: str = "M1"
    start: datetime | None = None
    end: datetime | None = None
    initial_equity: float = Field(default=100.0, gt=0)
    spread_mode: SpreadMode = "per_bar"
    spread_value: float | None = Field(default=None, ge=0)
    commission_per_lot_per_side: float = Field(default=0.0, ge=0)
    swap_mode: SwapMode = "points"
    session_threshold: float = Field(default=0.5, gt=0, le=1)


class StrategyRef(Model):
    """The strategy arrives by id (from `strategies/`) or inline."""

    strategy_id: str | None = None
    spec: dict[str, Any] | None = None


class BacktestRequest(StrategyRef):
    config: RunConfigIn
    force: bool = False


class BacktestResponse(Model):
    run_id: str
    status: RunStatus
    reused: bool
    message: str
    poll_url: str


class EdgeRequest(StrategyRef):
    config: RunConfigIn
    horizons: list[int] | None = None
    min_observations: int | None = Field(default=None, ge=2)


class EdgeStatOut(Model):
    horizon: int
    direction: Literal["long", "short", "both"]
    observations: int
    mean_points: float | None
    std_points: float | None
    stderr_points: float | None
    t_stat: float | None
    p_value: float | None
    drift_baseline_points: float | None
    spread_cost_points: float | None
    net_points: float | None
    t_vs_cost: float | None
    p_vs_cost: float | None
    beats_cost: bool
    verdict: str


class BreakevenPriorOut(Model):
    """A-priori break-even win rate, from the spec and cost assumptions alone."""

    valid: bool
    reason: str | None = None
    breakeven_win_rate: float | None = None
    loss_points: float | None = None
    win_points: float | None = None
    commission_points: float | None = None
    avg_spread_points: float | None = None
    caveats: list[str] = Field(default_factory=list)


class EdgeResponse(Model):
    symbol: str
    timeframe: str
    bars: int
    start: datetime | None
    end: datetime | None
    horizons: list[int]
    signals_long: int
    signals_short: int
    min_observations: int
    significance: float
    stats: list[EdgeStatOut]
    passed: bool
    verdict: str
    breakeven_prior: BreakevenPriorOut | None = None


# -- run -----------------------------------------------------------------


class MetricsOut(Model):
    label: str
    start: datetime | None
    end: datetime | None
    days: float
    initial_equity: float
    final_equity: float
    total_return: float | None
    annual_return: float | None
    sharpe: float | None
    sortino: float | None
    max_drawdown_money: float | None
    max_drawdown_pct: float | None
    calmar: float | None
    trades: int
    win_rate: float | None
    profit_factor: float | None
    expectancy: float | None
    avg_duration: float | None = Field(default=None, description="seconds")
    exposure: float | None
    max_losing_streak: int
    t_stat: float | None
    p_value: float | None
    r_multiples: dict[str, float | None] = Field(default_factory=dict)
    costs: dict[str, float | None] = Field(default_factory=dict)
    ambiguous_trades: int = 0
    gap_crossing_trades: int = 0
    text: str = ""
    drawdown_unsustainable: bool = False


class BreakevenOut(Model):
    """Break-even win rate computed from realized trades, or why it is invalid."""

    valid: bool
    reason: str | None = None
    observations: int
    breakeven_win_rate: float | None = None
    realized_win_rate: float | None = None
    delta: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    non_binary_trades: int = 0
    non_binary_reasons: dict[str, int] = Field(default_factory=dict)


class ExecutionOut(Model):
    signals_long: int
    signals_short: int
    trades: int
    ambiguous_trades: int
    gap_crossing_trades: int
    blocked: dict[str, int] = Field(default_factory=dict)
    exit_reasons: dict[str, int] = Field(default_factory=dict)


class RunSummaryOut(Model):
    run_id: str
    status: RunStatus
    created_at: datetime
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    data_start: datetime | None
    data_end: datetime | None
    bars: int
    duration_seconds: float | None
    error: str | None
    trades: int | None
    final_equity: float | None
    total_return: float | None
    sharpe: float | None
    max_drawdown_pct: float | None
    profit_factor: float | None
    win_rate: float | None
    ambiguous_trades: int | None


class RunDetailOut(Model):
    run_id: str
    status: RunStatus
    created_at: datetime
    finished_at: datetime | None
    duration_seconds: float | None
    engine_version: str
    spec_hash: str
    data_hash: str
    bars: int
    data_start: datetime | None
    data_end: datetime | None
    error: str | None
    spec: dict[str, Any]
    config: dict[str, Any]
    strategy: MetricsOut | None = None
    benchmark: MetricsOut | None = None
    breakeven: BreakevenOut | None = None
    execution: ExecutionOut | None = None


class EquityPoint(Model):
    t: datetime
    equity: float
    drawdown: float


class EquityOut(Model):
    run_id: str
    initial_equity: float
    total_points: int
    returned_points: int
    downsampled: bool
    points: list[EquityPoint]


class TradeOut(Model):
    index: int
    direction: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    exit_reason: str
    lots: float
    bars_held: int
    session_bars_held: int
    gross_pnl: float
    spread_points: float
    spread_cost: float
    commission: float
    swap: float
    net_pnl: float
    risk_money: float | None
    r_multiple: float | None
    ambiguous: bool
    crossed_gap: bool


class TradesOut(Model):
    run_id: str
    total: int
    offset: int
    limit: int
    ambiguous_total: int
    items: list[TradeOut]


# -- comparison ----------------------------------------------------------


class ComparePoint(Model):
    t: datetime
    values: list[float | None]


class CompareRunOut(Model):
    run_id: str
    label: str
    strategy_id: str
    symbol: str
    timeframe: str
    start: datetime | None
    end: datetime | None
    initial_equity: float


class CompareMetricRow(Model):
    key: str
    label: str
    format: Literal["pct", "money", "number", "int", "text"]
    higher_is_better: bool | None
    values: list[float | None]
    delta: list[float | None]


class CompareRequest(Model):
    run_ids: list[str] = Field(min_length=2)
    points: int = Field(default=1500, ge=50, le=5000)
    normalize: bool = True


class CompareConfigRow(Model):
    """One config field whose value differs between the compared runs."""

    key: str
    values: list[str | None]


class CompareResponse(Model):
    runs: list[CompareRunOut]
    normalized: bool
    axis_start: datetime | None
    axis_end: datetime | None
    series: list[ComparePoint]
    metrics: list[CompareMetricRow]
    config_diff: list[CompareConfigRow] = Field(default_factory=list)
    configs_identical: bool = True
    warnings: list[str] = Field(default_factory=list)


class DeleteResponse(Model):
    run_id: str
    deleted: bool


class ErrorResponse(Model):
    detail: str


# -- validation: walk-forward --------------------------------------------


class WalkForwardRequest(Model):
    run_id: str = Field(min_length=1)
    mode: Literal["rolling", "anchored"] = "rolling"
    train_days: int = Field(default=90, ge=1, le=3650)
    test_days: int = Field(default=30, ge=1, le=3650)
    min_train_trades: int = Field(default=30, ge=0)
    objective: Literal["net_pnl", "sharpe", "profit_factor", "expectancy"] = "net_pnl"
    grid: dict[str, list[Any]] | None = Field(
        default=None,
        description=(
            "dotted spec path -> values to try, e.g. "
            "exit.stop_loss.value: [100, 150, 200]"
        ),
    )


class ParameterGridIn(Model):
    """Standalone shape of a grid, so the frontend gets a named type."""

    grid: dict[str, list[Any]] = Field(default_factory=dict)


class WalkForwardMetricsOut(Model):
    """Metrics of one leg. Extra keys are allowed: the engine may add more."""

    model_config = ConfigDict(extra="allow")

    trades: int | None = None
    net_pnl: float | None = None
    total_return: float | None = None
    sharpe: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    expectancy: float | None = None
    max_drawdown_pct: float | None = None
    final_equity: float | None = None
    initial_equity: float | None = None
    start: datetime | None = None
    end: datetime | None = None


class WalkForwardWindowOut(Model):
    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    embargo_minutes: float
    skipped: bool
    skip_reason: str | None = None
    candidates_evaluated: int
    chosen_params: dict[str, Any] = Field(default_factory=dict)
    train_trades: int
    test_trades: int
    train_metrics: WalkForwardMetricsOut | None = None
    test_metrics: WalkForwardMetricsOut | None = None
    equity_start: float | None = None
    equity_end: float | None = None


class DegradationRowOut(Model):
    metric: str
    observations: int
    in_sample_mean: float | None = None
    out_of_sample_mean: float | None = None
    in_sample_stderr: float | None = None
    out_of_sample_stderr: float | None = None
    ratio: float | None = None
    ratio_note: str | None = None


class StabilityRowOut(Model):
    parameter: str
    observations: int
    distinct_values: int
    chosen: list[str]
    counts: dict[str, int]
    mode: str
    mode_share: float


class OosPointOut(Model):
    t: datetime
    equity: float


class WalkForwardResponse(Model):
    run_id: str
    symbol: str
    timeframe: str
    mode: Literal["rolling", "anchored"]
    train_days: int
    test_days: int
    min_train_trades: int
    objective: str
    embargo_minutes: float
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    grid_size: int
    optimized: bool
    windows: list[WalkForwardWindowOut] = Field(default_factory=list)
    windows_evaluated: int
    windows_skipped: int
    oos_trades: int
    oos_metrics: WalkForwardMetricsOut | None = None
    oos_curve: list[OosPointOut] = Field(default_factory=list)
    degradation: list[DegradationRowOut] = Field(default_factory=list)
    parameter_stability: list[StabilityRowOut] = Field(default_factory=list)
    verdict: str
    warnings: list[str] = Field(default_factory=list)


# -- validation: permutation ---------------------------------------------


class PermutationRequest(Model):
    run_id: str = Field(min_length=1)
    iterations: int = Field(default=1000, ge=10, le=20000)
    tests: list[Literal["random_entries", "permuted_returns"]] = Field(
        default_factory=lambda: ["random_entries", "permuted_returns"], min_length=1
    )
    block_bars: int = Field(default=512, ge=2, le=100000)
    seed: int = 12345


class PermutationStatisticOut(Model):
    key: str
    label: str
    observed: float
    null_mean: float
    null_std: float
    null_p05: float
    null_p50: float
    null_p95: float
    percentile: float
    p_value: float
    iterations: int
    better_than_null: bool


class HistogramOut(Model):
    edges: list[float]
    counts: list[int]
    observed: float


class PermutationTestOut(Model):
    kind: Literal["random_entries", "permuted_returns"]
    description: str
    iterations: int
    iterations_requested: int
    seed: int
    elapsed_seconds: float
    observed_trades: int
    null_trades_mean: float
    null_trades_std: float
    statistics: list[PermutationStatisticOut] = Field(default_factory=list)
    histogram: HistogramOut | None = None
    verdict: str
    warnings: list[str] = Field(default_factory=list)
    block_bars: int | None = None


class PermutationResponse(Model):
    run_id: str
    symbol: str
    tests: list[PermutationTestOut] = Field(default_factory=list)


# -- validation: multiple testing ----------------------------------------


class MultipleTestingRequest(Model):
    run_id: str = Field(min_length=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    grid: dict[str, list[Any]] | None = Field(
        default=None,
        description="optional grid; without it PBO is declared not computed",
    )


class TrialOut(Model):
    run_id: str
    strategy_id: str
    spec_hash: str
    trades: int
    sharpe_per_trade: float
    net_pnl: float
    p_value: float | None = None


class DeflatedSharpeOut(Model):
    valid: bool
    reason: str | None = None
    observed_sharpe_per_trade: float | None = None
    observed_sharpe_annualized: float | None = None
    expected_max_sharpe: float | None = None
    deflated_sharpe: float | None = None
    probabilistic_sharpe: float | None = None
    trials: int
    observations: int
    skewness: float | None = None
    kurtosis: float | None = None
    variance_across_trials: float | None = None


class ThresholdRowOut(Model):
    run_id: str
    strategy_id: str
    p_value: float
    rank: int
    bonferroni_threshold: float
    benjamini_hochberg_threshold: float
    passes_bonferroni: bool
    passes_benjamini_hochberg: bool
    mean_pnl_positive: bool


class PBOOut(Model):
    valid: bool
    reason: str | None = None
    pbo: float | None = None
    splits: int = 0
    slices: int = 0
    candidates: int = 0
    logits: list[float] = Field(default_factory=list)


class MultipleTestingResponse(Model):
    run_id: str
    symbol: str
    period_start: datetime | None = None
    period_end: datetime | None = None
    trials: int
    trials_detail: list[TrialOut] = Field(default_factory=list)
    alpha: float
    deflated_sharpe: DeflatedSharpeOut
    thresholds: list[ThresholdRowOut] = Field(default_factory=list)
    pbo: PBOOut
    verdict: str
    warnings: list[str] = Field(default_factory=list)


# -- validation: tick resolve --------------------------------------------


class TickResolveRequest(Model):
    run_id: str = Field(min_length=1)


class ResolvedTradeOut(Model):
    index: int
    direction: int
    entry_time: datetime
    exit_time: datetime
    original_reason: str
    resolution: Literal["stop_loss", "take_profit", "unresolved"]
    reason: str | None = None
    ticks: int
    stop_level: float
    target_level: float
    original_net_pnl: float
    resolved_net_pnl: float
    delta: float


class TickResolveResponse(Model):
    run_id: str
    symbol: str
    available: bool
    reason: str | None = None
    ambiguous_trades: int
    resolved: int
    unresolved: int
    resolved_to_take_profit: int
    resolved_to_stop_loss: int
    trades: list[ResolvedTradeOut] = Field(default_factory=list)
    original_net_pnl: float
    resolved_net_pnl: float
    delta_net_pnl: float
    original_final_equity: float | None = None
    resolved_final_equity: float | None = None
    original_win_rate: float | None = None
    resolved_win_rate: float | None = None
    verdict: str
    warnings: list[str] = Field(default_factory=list)


# -- batch ---------------------------------------------------------------


class BatchPeriodIn(Model):
    start: datetime | None = None
    end: datetime | None = None


class BatchRequest(StrategyRef):
    symbols: list[str] = Field(min_length=1, max_length=64)
    config: RunConfigIn
    periods: list[BatchPeriodIn] = Field(default_factory=list)
    grid: dict[str, list[Any]] | None = None
    consistency_metric: Literal["mean_r", "expectancy", "net_pnl", "sharpe"] = "mean_r"
    max_workers: int | None = Field(default=None, ge=1, le=64)


class BatchRunOut(Model):
    symbol: str
    params: dict[str, Any] = Field(default_factory=dict)
    period_start: datetime | None = None
    period_end: datetime | None = None
    run_id: str | None = None
    status: str
    error: str | None = None
    bars: int
    trades: int
    net_pnl: float | None = None
    expectancy: float | None = None
    expectancy_stderr: float | None = None
    mean_r: float | None = None
    mean_r_stderr: float | None = None
    sharpe: float | None = None
    profit_factor: float | None = None
    win_rate: float | None = None
    max_drawdown_pct: float | None = None
    final_equity: float | None = None
    p_value: float | None = None


class ConsistencyOut(Model):
    metric: str
    observations: int
    positive: int
    negative: int
    zero_or_missing: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    stderr: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    sign_p_value: float | None = None
    carried_by_one_symbol: bool
    dominant_symbol: str | None = None
    verdict: str


class BatchResponse(Model):
    strategy_id: str
    symbols: list[str]
    periods: list[BatchPeriodIn] = Field(default_factory=list)
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    grid_size: int
    cells: list[BatchRunOut] = Field(default_factory=list)
    completed: int
    failed: int
    elapsed_seconds: float
    consistency: ConsistencyOut
    verdict: str
    warnings: list[str] = Field(default_factory=list)
