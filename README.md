# backtest-engine

Backtesting engine for trading strategies on MetaTrader 5 data.

There is not a single line of code here that can send an order. The project
exists to answer one question honestly: *given these bars, these costs and
this spec, what would have happened — and how much of that is statistically
meaningful?*

## Requirements

- Windows, Python 3.10+
- MetaTrader 5 terminal installed, running and logged in (for downloads;
  cached data works offline)

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

## Layout

```
core/data/provider.py      abstract DataProvider + SymbolSpec + Timeframe
core/data/mt5_provider.py  MetaTrader 5 implementation (read only)
core/data/servertime.py    server time <-> UTC conversions (pure, testable)
core/data/cache.py         Parquet cache per (symbol, timeframe, year) + JSON metadata
core/data/quality.py       quality report: gaps, duplicates, NaN, malformed bars
core/data/hc_reader.py     decoder for MT5's .hc cache (offline history)
core/indicators/           pure functions + name -> function registry
core/strategy/             pydantic spec, bar features, evaluator
core/engine/               costs, sizing, risk gates, backtester
core/metrics/              performance, break-even win rate, buy & hold benchmark
core/research/edge.py      gate zero: does the signal beat the spread?
core/runs/                 run store, orchestration, golden snapshots
core/validation/           walk-forward, permutation, multiple testing, tick resolve
core/batch/                same spec over many instruments + cross-sectional consistency
core/version.py            engine version (single source of truth)
api/                       FastAPI, 127.0.0.1 only
ui/                        Vite + React + TypeScript
strategies/                the JSON specs
tests/                     pytest; integration tests are marked `mt5`
scripts/run_batch.py       batch runner driven by a YAML file
run.py                     starts everything and opens the browser
```

`provider.py` contains nothing MetaTrader-specific: adding a second provider
means implementing four methods and honoring the DataFrame contract.

## Data contract

- `get_bars` returns a **tz-aware UTC** `DatetimeIndex`, sorted, without
  duplicates, with columns `open, high, low, close, tick_volume, spread,
  real_volume`.
- `spread` is in points and always preserved: it is the real transaction
  cost, not an accessory detail. On XAUUSD.r in 2025 the median is 7 points,
  i.e. 0.07 USD.
- `get_ticks` returns `bid, ask, last, volume`, same index type.
- Bounds are `[start, end)`: `copy_rates_range` includes the right endpoint,
  which is trimmed on the way out so the cache coverage stays consistent.

## The critical point: the server timezone

MT5 exposes timestamps as epoch seconds that actually encode **the server's
wall clock**, not a UTC instant. If the server clock shows 13:44 and sits at
UTC+3, the value received is the epoch of 13:44 UTC. Taking it at face value
shifts the whole history by 2-3 hours, which on an intraday strategy is
enough to invalidate any result.

The timezone is **not hardcoded**. It is measured by comparing the last tick
with the local clock, quantizing to 30 minutes (so network latency stays out
of the count), and resolving the offset to an IANA zone among the candidates
in `servertime.DEFAULT_CANDIDATE_ZONES`. Using an IANA zone rather than a
fixed offset reconstructs past DST transitions correctly. If no candidate
matches, a fixed offset is used and declared in the logs.

Two distinct traps, both covered by tests:

1. **Outbound**: epochs must be read as naive and localized on the server
   timezone.
2. **Inbound**: the `MetaTrader5` package converts **naive** `datetime`s
   using the *local machine's* timezone. On a PC not set to UTC the request
   bounds arrive silently shifted (verified: 2 hours on a UTC+2 machine).
   That is why `utc_to_broker_datetime` always passes a tz-aware UTC datetime
   carrying the server's wall clock.

With the market closed the last tick is too old to measure the offset: in
that case `MT5Provider` raises `ServerTimeError`. The timezone detected
during the last download stays in the cache metadata and can be passed back
with `server_timezone=`.

## Warm-up

Before every deep historical request the provider calls
`copy_rates_from_pos(symbol, tf, 0, 10)`. Without it the terminal only
answers with the bars already in memory and long `copy_rates_range` calls
come back empty.

## Cache

`<root>/<symbol>/<timeframe>/<year>.parquet`, with `<year>.json` next to it:

```json
{
  "schema_version": 1,
  "symbol": "XAUUSD.r",
  "timeframe": "M1",
  "year": 2025,
  "coverage_utc": [["2025-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"]],
  "rows": 265609,
  "first_bar_utc": "2025-04-02T04:44:00+00:00",
  "last_bar_utc": "2025-12-31T21:59:00+00:00",
  "downloaded_at_utc": "2026-08-31T10:45:54.751926+00:00",
  "server_timezone": "Europe/Athens"
}
```

Parquet and not pickle: readable by other tools and stable across pandas
versions.

Coverage is a **list of intervals**, not a single range: two downloads far
apart in time must not make the hole in between pass for "already
downloaded". `get_or_fetch` downloads only the missing holes.

A requested interval where the broker has no data still counts as covered —
as in the example above, where the broker's history starts on April 2nd but
the coverage spans all of 2025. This is deliberate, otherwise every run
would re-download the void. When the broker extends the history backwards,
an explicit invalidation is needed: `cache.invalidate(symbol, timeframe,
year)`, or `--refresh` in the example script.

## Data quality

`check_quality` measures and reports, **it corrects and discards nothing**.
The session calendar is derived from the data: for each weekly slot, count in
how many weeks that minute is quoted; slots below the threshold count as
closed. Nobody writes "weekend" anywhere, so the same logic holds for
instruments with different hours. The report also declares the calendar's
confidence: below 4 weeks of history it is worthless.

What is reported: in-session gaps (with start, end and missing bars),
duplicate timestamps, unsorted index, NaN per column, `high < low`, OHLC
outside its own range, non-positive prices, `tick_volume == 0`,
`spread == 0`, plus a few sample timestamps per anomaly.

## Example: a year of M1 with quality report

```
.venv\Scripts\python -m examples.download_year XAUUSD.r --timeframe M1 --year 2025
```

The symbol is mandatory: there is no default one in the code. Options:
`--cache` (directory, default `data_cache/`), `--refresh` (invalidate the
year before downloading), `--log-level`.

Real output on this installation (fpmarkets, server Europe/Athens): 265,609
M1 bars from 2025-04-02 to 2025-12-31, 6,893/10,080 active weekly slots,
97.46% completeness, 83 gaps (holidays and early closes), median spread 7.0
points. Read it for what it is: the broker exposes **9 months**, not 12, and
2.5% of the session bars are missing. Whoever uses this data for a backtest
must know that before, not after.

## Strategies and engine

A strategy is a **JSON file**, not code. The engine validates it, evaluates
it vectorized and executes it bar by bar with explicit fill rules.

### Indicators

`sma, ema, rsi, atr, bollinger, macd, stoch, donchian, roc`, with a registry
by name.

**No fillna on warm-up.** The first bars stay NaN and propagate to the
signals as "no signal". An `rsi.fillna(50)` fabricates crossings that never
existed, right at the start of the sample where nobody looks. The tests
verify the exact warm-up length of every indicator.

Single-series functions have signature `(series, **params)`. ATR, stochastic
and Donchian take `high`/`low`/`close` as explicit arguments: the true range
is not computed on the close alone, and pretending otherwise for signature
uniformity would be a bug dressed as design. Multi-output indicators are
referenced as `id.output` (`mac.histogram`).

### Spec

Pydantic v2 with `extra="forbid"`. Validation checks unique ids, `ref`s
pointing to existing indicators, correct outputs for multi-output
indicators, parameters consistent with the type. An error lists every
problem with the field path.

`stop_loss`/`take_profit` accept only `type: "points"`; `risk.news_filter`
must be `null` (it arrives with the live runner). A declared but never
referenced indicator produces a warning, not an error.

### Evaluator

Recursive dispatch over pydantic models: **no `eval`, no `exec`**. A spec is
data, not code to execute.

Operators: `and, or, not, gt, gte, lt, lte, eq, between, cross_above,
cross_below, rising, falling`. Operands: `{"ref": id}`, `{"const": n}`,
`{"bar": field}`, `{"feature": name}`. A NaN input makes the condition
false, never accidentally true. `eq` between floats uses `np.isclose`.

Bar features: `lower_wick_ratio`, `upper_wick_ratio`, `body_ratio`,
`range_points`, `close_position_in_range`. On zero-range bars the ratios are
`0/0` and stay NaN.

### The look-ahead test

`tests/test_lookahead.py` is the test that holds up everything else. For 500
sampled bars it recomputes the signals on `bars.iloc[:t+1]` — a world where
the future does not exist — and demands the same value as the vectorized
computation on the whole DataFrame. It runs on two specs: the baseline and
one touching every operator and operand type. A twin test does the same at
the indicator level, to localize the culprit instead of just knowing that
something peeks.

### Costs

Everything parametric on `SymbolSpec`: **no per-point value written in the
code**.

- **Spread** from the feed's `spread` column, bar by bar. Alternative
  policies `fixed` and `quantile` for stress tests.
- **Commission** per lot per side.
- **Swap** at every **server** midnight crossed (timezone from the data
  layer), triple on the Wednesday-to-Thursday night. `SymbolSpec` does not
  expose `swap_mode`, so the interpretation (`points` or `money`) is
  declared in `SwapModel` rather than guessed.
- **Conversion**: `(point / tick_size) * tick_value * lots`. MT5's
  `tick_value` is in the **account** currency — on this installation the
  account is in EUR and gold quotes in USD, which is why it is ~0.86 and not
  1.00. If missing, the fallback is `point * contract_size`, which however
  gives the profit currency: the fallback is logged.

`gross_pnl` is the PnL **at zero spread**, `spread_cost` the spread actually
paid, `net_pnl = gross - spread - commission + swap`. The decomposition is
exact and tested: the same signal with real costs and with
`CostModel.zero()` differs by exactly spread + commission.

### Execution rules

1. Signal from the **close** of t, execution at the **open** of t+1.
2. MT5 bars are **bid**. A BUY enters at `open + spread` (ask) and exits on
   the bid; a SELL enters on the bid and exits on the ask. Touch tests use
   the same side of the book: a short's stop is evaluated on the ask.
3. **Gaps**: a bar opening beyond the stop -> fill at the **open price**,
   not the level. Same for the target. Filling at the level truncates the
   loss tail by construction: in the test the difference between the two
   conventions is -51.00 vs -14.00.
4. Stop and target touched within the same bar -> the **stop** is assumed,
   the trade is flagged `ambiguous`. Without tick data there is no way to
   know which came first: the count must be watched.
5. **Time stop in session bars**, not array rows: the session calendar also
   counts the bars missing from the data. Trades crossing a hole are flagged
   `crossed_gap`.
6. The `risk.py` gates (open positions, max spread, cooldown, trades per
   day, session) are written to be used **identically** by the live runner.

The engine holds **one position at a time**: `max_open_positions > 1` raises
`NotImplementedError` instead of pretending.

`min_lot` is a **floor**, not an entry threshold: below one equity step the
minimum lot is traded anyway. The per-trade percentage risk rises as the
account shrinks — a property of this sizing, not a bug.

### Metrics

Sharpe and Sortino annualized with a factor **derived from the session
calendar** of the data (active weekly slots x 52.18), not a 252 or a 365
dropped in. Risk-free rate 0. Plus absolute and percentage max drawdown,
Calmar, profit factor, expectancy, win rate, average duration, exposure,
R-multiple distribution, max losing streak, t-stat and p-value on the mean
per-trade return.

Every report is shown next to the **buy & hold** over the same period, with
the same costs. If the drawdown exceeds 100% the report says so explicitly:
those metrics are arithmetic, not an achievable result.

### Break-even win rate

When every trade ends on either the stop or the target, the strategy is a
binary bet and the win rate needed to break even is arithmetic:
`avg_loss / (avg_loss + avg_win)`. The engine computes it from the realized
trades and shows it next to the realized win rate, with the delta. If the
outcome distribution is **not** binary (trailing stops, signal exits,
frequent time stops), the number is not shown and the reason is stated.

An a-priori estimate — from the spec's SL/TP and the average spread alone —
is part of the gate zero report. Note on the spread: in this engine SL/TP
levels are anchored to the spread-inclusive entry price, so the spread does
not change the per-trade cash amounts; it shifts the trigger levels, which
lowers the *achievable* win rate instead of raising the threshold.

## Gate zero

Runs **before** the backtest and answers a single question: after the
signal, does the price move in the predicted direction more than it would
after a random entry, and by enough to cover the spread?

If the answer is no, no SL/TP combination can rescue the signal: SL and TP
redistribute returns across trades, they do not create them. Searching for
the right SL/TP grid on a signal without an edge is the most efficient way
to find noise and call it a discovery.

For each horizon and direction it measures: observations, mean return in
points, standard deviation and standard error, t-stat and p-value against
zero, the **drift baseline** (the same mean return over all bars, weighted by
the long/short imbalance) and the **average spread** on the signal bars. The
verdict is explicit: is the net (mean − drift − spread) positive with
significance, yes or no.

Drift is not a detail: on an instrument that rose 40% in the period, a
mostly-long signal "profits" with no merit of its own. A dedicated test
verifies that a trend is not mistaken for an edge.

## Persisted runs

`runs/<run_id>/` with `spec.json`, `config.json`, `trades.parquet`,
`equity.parquet`, `metrics.json`, `meta.json`. They open with an editor and
read with pandas.

`run_id` is **deterministic**: a hash of spec + configuration + data
fingerprint. Relaunching the same thing neither recomputes nor duplicates;
changing a parameter, the spread or a single candle produces a different id,
and the old run stays there documenting how things were. The data
fingerprint hashes the bar bytes, not the dates: if the broker rewrites a
candle, the time range does not notice — the hash does.

## Engine versioning and the golden test

The engine version lives in `core/version.py` and follows semantic
versioning from the point of view of run results (see
`ENGINE_CHANGELOG.md`). The golden test (`tests/golden/`) pins the complete
trade list and all metrics of the baseline strategy on a frozen data window;
any silent change in the results fails the suite. To update the reference
after an intentional change:

```
python -m scripts.update_golden --update-golden --note "reason"
```

The note is mandatory and lands in `ENGINE_CHANGELOG.md`. The reference also
pins the `SymbolSpec` (tick_value moves with FX rates) and the server
timezone, so the test never depends on a live terminal.

## Statistical validation

A backtest number is a hypothesis, not a result. Four independent attacks on
it live in `core/validation/`, and each one is allowed to say "this proves
nothing" — which on the baseline strategy is exactly what most of them say.

### Walk-forward

Rolling or anchored windows. On each in-sample leg an optional parameter grid
is optimized; the winner is applied to the leg that follows, and the process
steps forward. **Only the concatenated out-of-sample curve may be read as
performance**: every in-sample figure is the best of N candidates chosen on
that very data and is an upper bound by construction.

Three decisions that make the difference between a walk-forward and a
decoration:

- **Windows are sized in trades, not just days.** An in-sample leg holding
  fewer than `min_train_trades` (default 30) is discarded and reported as
  discarded — optimizing on eleven trades selects the candidate that got
  lucky eleven times.
- **An embargo separates the legs**, as wide as the longest holding the spec
  can produce (its time stop, or the longest holding actually observed).
  Without it a trade open across the boundary is scored on both sides of it.
- **Equity is carried across windows.** Sizing depends on equity, so
  restarting each window at the initial capital would measure a strategy
  nobody could have traded.

`tests/test_walkforward.py` mutates the price series past a window's test end
and asserts every earlier window comes back bit-identical: chosen parameters,
trade counts and metrics. That is the test that would catch leakage.

### Permutation

Two null models, both executed through the real `Backtester` — never a
simplified re-implementation, because a null that fills differently measures
the difference between two simulators.

- **random_entries** — same exit rules, costs, gates and long/short signal
  counts, but the entry timestamps are random. Answers: does the entry rule
  know anything the calendar does not?
- **permuted_returns** — same strategy on a price path rebuilt by resampling
  blocks of consecutive returns. Autocorrelation and volatility clustering
  survive inside a block; the specific sequence of events does not. Each bar
  keeps its own shape (open, high and low travel as log offsets from their
  own close), so a resampled bar is still a valid bar.

The p-value is `(1 + #{null >= observed}) / (1 + N)`: it cannot reach zero
however many draws are taken. 1000 iterations over nine months of M1 take
about 130 s and 150 s respectively on 8 worker processes; the pool is capped
because each worker holds its own copy of the bar series.

### Multiple testing

The trial count comes from the run store — distinct spec hashes over
identical data — plus the grid candidates when a grid is passed, since a
candidate is an attempt exactly like a saved run is.

- **Deflated Sharpe Ratio** (Bailey & López de Prado) on the **per-trade**
  Sharpe, never the annualized one: annualizing multiplies by
  `sqrt(periods)` and would silently inflate the correction. Reports the
  Sharpe the search buys for free (the expected maximum of N trials), the
  skew and kurtosis correction, and the resulting probability that the true
  Sharpe is above zero.
- **Bonferroni and Benjamini-Hochberg**, always both. They disagree exactly
  when the answer is delicate, and hiding that disagreement is how a
  screening result gets sold as a confirmation.
- **PBO via CSCV** when a grid is supplied; when one is not, the report says
  it was not computed rather than returning a number.

Every p-value here is two-sided on mean trade PnL, so on a losing strategy a
small p means "reliably losing". The sign travels next to the p-value in
every row for that reason.

### Tick resolution of ambiguous trades

When one bar touches both stop and target the engine assumes the stop. That
is the right default and it is still an assumption. `tick_resolve` pulls the
ticks of the exit bar and asks which level came first, reading a long on the
bid and a short on the ask exactly as the engine prices exits, then reports
the delta against the conservative assumption.

Missing ticks, a closed terminal, or a period older than the broker's tick
history are reported as unresolved and counted. Nothing is assumed in their
place, and an unresolvable run is stated as untested rather than confirmed.

## Batch across instruments

One instrument is one experiment. Seventy-two long trades over nine months
carry a standard error wide enough to swallow any edge of this size, so a
single-symbol result is not evidence in either direction.

```
python -m scripts.run_batch batch.yaml --json report.json
```

```yaml
strategy: strategies/rsi-wick-baseline.json
symbols: [XAUUSD.r, XAGUSD.r, EURUSD.r]
timeframe: M1
initial_equity: 100
spread_mode: per_bar
consistency_metric: mean_r
periods:
  - {start: 2025-04-02, end: 2025-08-01}
grid:
  exit.stop_loss.value: [100, 150, 200]
```

`periods` and `grid` are optional. Every cell is a normal run with the same
deterministic id and lands in `runs/`; the batch orchestrates, it does not
re-implement the engine. A cell that fails (usually: no cached data for that
instrument) is recorded with its reason and excluded from the aggregate,
which says how many were lost.

The output that matters is the **cross-sectional consistency** block: on how
many instruments the metric has the same sign, a binomial sign test against a
fair coin, the spread across instruments, and a flag when removing the single
largest instrument flips the aggregate — the signature of one instrument's
history rather than a strategy property. `mean_r` is the default metric
because the R multiple is normalized by the trade's own risk and stays
comparable across instruments whose point values differ by orders of
magnitude.

A caveat the tool cannot fix for you: a stop expressed in **points** is not
the same distance on two instruments. 150 points is 0.04% of price on gold
and 0.14% on EURUSD, so a cross-instrument batch of a point-based spec is
testing a family of related strategies, not one strategy ten times.

## API

`127.0.0.1` only, CORS open only to `localhost`, no authentication because
it never leaves the machine. Endpoints: symbols and data coverage,
strategies and their validation, gate zero, backtest, runs (listing, detail,
equity, trades, comparison, deletion), statistical validation
(`/api/validation/walkforward`, `/permutation`, `/multiple-testing`,
`/tick-resolve`) and `/api/batch`.

- The backtest runs in a thread pool. Beyond **2 seconds** the request still
  returns the `run_id` with status `running` and the frontend polls
  `GET /api/runs/{id}`. A run stuck beyond the hard limit is declared failed
  on the first read — a thread cannot be killed in Python, and this README
  does not pretend otherwise.
- `GET /api/runs/{id}/equity` reduces the curve to ~2000 points with LTTB.
  The **drawdown is computed on the whole series** and only then are points
  reduced: the minimum must not vanish in the pruning. The max-drawdown and
  max-equity points are forced into the sample.
- Errors leave with the right status and a readable sentence: 409 if the
  data is not cached, 422 for an invalid spec, 503 if the MT5 terminal does
  not respond, 404 if the run does not exist. Never a stack trace in the
  browser.
- `inf` and `NaN` are not valid JSON: they come out as `null` (an infinite
  profit factor is data, but `JSON.parse` rejects it).
- `POST /api/runs/compare` also returns the **diff of the run
  configurations**: only the fields that differ, generically over any config
  field, so two runs differing only in cost policy are labeled as such.

## Dashboard

Vite + React + TypeScript. **TypeScript types are generated from the
OpenAPI schema** (`python -m scripts.export_openapi` + `npm run gen:api`),
not written by hand: a field renamed in the API breaks the UI build instead
of becoming an `undefined` on screen. `run.py` regenerates them by itself
when the schema changes.

The frontend **computes no metric**: drawdown, curve normalization,
comparison deltas and column formats come from the API. The UI only decides
how many digits to show.

Pages:

- **Run** — form with symbol (and which have cached data), timeframe, period
  with the cache bounds shown next to the fields, cost policy and
  commission. Two buttons: "Check edge" (fast) and "Run backtest".
- **Result** — equity and drawdown aligned on the same axis, metrics next to
  the buy & hold with the unsustainable-drawdown note, break-even win rate
  next to the realized one, execution (signals, exits, gates) and paginated
  trades with the ambiguous ones highlighted.
- **Compare** — two or more runs overlaid, curves normalized to 100, metrics
  side by side with the delta against the first and the "better" direction
  declared per metric. The config fields that differ between the runs are
  shown under each column and as rows; identical configs are declared as
  such. Warns if the runs are on different instruments or periods.
- **Validation** — walk-forward (concatenated OOS curve, per-window table,
  IS/OOS degradation with standard errors, parameter stability), permutation
  with the null histogram and the real strategy marked on it, multiple
  testing with observed and deflated Sharpe side by side, and tick
  resolution of the ambiguous trades.
- **Batch** — the same spec across instruments, a sortable symbol x metric
  table with the standard error next to each estimate, and the
  cross-sectional consistency row in the lead position.

Non-negotiable choices applied: tabular digits everywhere, **times always in
UTC and labeled as such** (the MT5 server clock is Europe/Athens, which is
neither UTC nor local time), color never the only carrier of meaning (sign,
badge or label next to it), loading states with reserved height, actionable
empty states, errors that say how to fix, visible keyboard focus,
`prefers-reduced-motion` respected, no native popups, no component library.

## Running

```
python run.py
```

Exports the schema, regenerates types if needed, starts uvicorn and Vite,
**waits until they actually respond** (not "wait five seconds and hope"),
opens the browser. Ctrl+C stops everything. Options: `--no-browser`,
`--no-ui`, `--reload`, `--api-port`, `--ui-port`.

No API keys, no external services, no telemetry: everything stays on the
machine.

## Tests

```
.venv\Scripts\python -m pytest
```

Integration tests are marked `mt5` and skipped if the terminal does not
respond; they use the first Market Watch symbol, or the one in
`MT5_TEST_SYMBOL`. The suite covers round-trip timezone conversion (DST
included), cache merging with holes, gap detection on synthetic data, the
look-ahead test, hand-computed engine fills, the golden reference, and a
check that the provider module contains no reference to order sending.

## Offline history (.hc)

`hc_reader` decodes the terminal's `.hc` cache when `copy_rates_*` is
unavailable (terminal not logged in). The file is locked by the terminal: it
must be copied by opening it read-only with sharing, not with `copy`.
Timestamps are in server time as with `copy_rates_*`: `read_hc_utc`
localizes them.
