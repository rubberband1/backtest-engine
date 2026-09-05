# backtest-engine

[![CI](https://github.com/rubberband1/backtest-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/rubberband1/backtest-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-2a6db2)](https://mypy-lang.org/)
[![Linted with ruff](https://img.shields.io/badge/ruff-clean-d7ff64)](https://docs.astral.sh/ruff/)


**A backtesting engine for MetaTrader 5 data, built so that it cannot flatter
a strategy. Across two campaigns it has screened 600 configurations — 10
classic strategies over 10 instruments and up to six years of history — and
reported that none of them survives the correction for having tried that many
things. That is the result, and the engine is what makes it trustworthy.**

Seventy of those 600 cells are refused before anything is tested on them,
because the broker's spread is too large a share of the intended stop, and a
refusal is not an experiment. That leaves **530 attempts**. The best cell with
enough trades to mean anything scores a Sharpe per trade of **+0.3307** over
32 trades — `ma-crossover` on `XTIUSD` H1 over 2025, charged a spread measured
on 74.9% of its bars. A search of 530 attempts is expected to reach **+0.3024**
by luck alone, and the observed value would have to clear **+0.6254** to be
credible at 95% confidence. It does not. Zero cells survive Bonferroni. The
engine says so in one line, on screen, above the results table.

**That campaign is in the repository**, in [campaigns/](campaigns/): the grid,
the frozen instrument specs, the report and the table, for both halves. One
command re-runs either from its manifest and diffs it cell by cell, and both
reproduce with every compared field matching. Every figure in this paragraph
names a cell in that report.

The most valuable thing it has produced is not a strategy but a correction to
its own arithmetic. Measured against the M1 bars inside them, **the `spread`
column of every bar above M1 turns out to be the *minimum* spread inside that
bar** — matching in 100.0% of four thousand hours on each of three
instruments. Charged as a transaction cost it bills the best price of the
period: 4 points instead of 29 on WTI at H1, and exactly zero on four FX
instruments, where 67–76% of hourly bars carry a zero spread. Every backtest
run that way had been trading for free on most bars. The engine now measures
the spread where the field means something, states what each run actually
charged, and refuses to call a pair testable when the broker's cut is too
large a share of the stop — before any strategy is run on it.

*Nothing sends an order unless you ask twice.* The research half of the
codebase cannot trade at all; the live runner defaults to dry run, refuses a
non-demo account when told to send, and shares its execution engine with the
backtester — a blocking test replays history through it and demands the same
trades, to the timestamp and the price.

### What that costs, in engineering

Getting to an honest "no" takes more machinery than getting to a hopeful
"yes":

- **Every number carries the observations it rests on and its standard
  error.** A test without statistical power says so instead of returning a
  figure.
- **A campaign counts its own attempts.** Three hundred backtests are three
  hundred chances to be lucky: at the 5% level about fifteen come back
  "significant" with no edge anywhere in the data. The trial count covers
  every cell that was *started*, including those stopped before a backtest.
- **The run's own inputs are pinned.** Instrument specs drift — the same
  batch once returned XTIUSD −21.05 and −19.86 under identical hashes because
  the broker had changed its swap rates in between — so the cost fields are
  part of a run's identity and are stored with it.
- **Assumptions that could decide the answer are reported as bands.** When a
  bar touches stop and target together the engine assumes the stop; on the
  baseline that assumption was worth twelve points of final equity, the
  difference between −10% and +1.5%. Every run reports both ends of it.
- **Silent filters are made loud.** Risk gates report what share of signals
  they rejected, on every row; one gate was quietly discarding half of them.
  When a run stops trading because the account fell below the broker's
  minimum lot, it says that too, rather than presenting a busted account as a
  cautious one.
- **A golden test pins the baseline** to the trade, so a change in results
  has to be declared rather than discovered later.
- **The simulator and the trader are the same code.** The execution rules
  live in one state machine; the backtester drives it over a DataFrame and
  the live runner drives it bar by bar. A replay test demands identical
  trades and fails the build otherwise.
- **A campaign freezes its own inputs.** `tick_value` follows an FX rate and
  moves while a campaign runs, so every cell used to be executed against a
  slightly different contract and re-running one never reproduced it. A
  campaign now writes a manifest, and one command re-runs it from that and
  diffs cell by cell. The case for it turned out to be stronger than the
  `tick_value` drift it was built for: between the previous campaign and the
  committed one the broker rewrote **XTIUSD's swap from −126.4 points a night
  to −0.7**, and the ten cells whose trade count changed are all on the two
  instruments whose swap moved. One of them went from a busted account at 21
  trades to +350 at 38. Without frozen contracts that is indistinguishable
  from an engine regression.
- **The spread says whether it was measured or assumed.** Where no M1 bars
  exist, the cost charged is a constant taken from a later period. Every run
  and every campaign cell now reports the share of its bars that had an M1
  sample behind them, and no result above is quoted without it. Of the 101
  cells that produced a Sharpe worth comparing, **33 had none at all**: the
  best of them, `rsi-mean-reversion` on `AUDUSD.r` H4 at +0.1990 over 172
  trades, was charged an assumed cost on all 8,214 of its bars.
- 596 tests, including one that recomputes signals on truncated history to
  prove no rule can see the future, one that demands a dry-run diary compare
  to its own backtest at exactly zero, and one that regenerates the shipped
  dataset and diffs it against what is committed.

Findings are written down even when they contradict the reason the feature
was built: normalizing exits on ATR was supposed to make instruments
comparable, and measurement showed it made the dispersion *worse* — that is
in the docs next to the feature. The same goes for mistakes in the statistics
themselves. The deflated-Sharpe correction was, for one campaign, fed the
per-trade Sharpe of cells with two trades; those reach ±10 by arithmetic
alone and pushed the corrected threshold to an unreachable +4.77. A threshold
nothing can clear is not a strict test, it is a broken one, and the changelog
says so.

## Quick start

Python 3.10 or newer, and Node 20+ for the dashboard. Nothing else: **no
MetaTrader 5 terminal, no broker account, no network.**

```
git clone https://github.com/rubberband1/backtest-engine
cd backtest-engine
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # Linux/macOS: .venv/bin/python
.venv/Scripts/python run.py
```

The browser opens on the dashboard with data already in it. The repository
ships a small **synthetic** dataset (`fixtures/data_cache/`, two invented
instruments over 2022-2023 on M1/H1/H4/D1), and `run.py` serves it whenever
`data_cache/` is empty — which is the case on a fresh clone.

> **The fixture data is invented.** It is a random walk with a plausible
> spread, session week and volume profile bolted on. Every metric computed on
> it — Sharpe, drawdown, p-value — describes a random number generator. The
> dashboard says so in a banner on every page, the API says so in
> `/api/health`, and the cache provenance records `synthetic_fixture` on
> every file. It exists so the engine can be run and read, not so anything
> can be concluded. Nothing in the results reported in this README was
> measured on it.

### Which numbers below a clone can check, and which it cannot

Worth stating plainly, because a README full of measurements should say where
each one comes from.

- **The campaign** — every figure in the opening paragraph — is committed in
  [campaigns/](campaigns/) as a report you can read without running anything.
  Re-executing it needs the bars, which are not in the repository.
- **The engine's behaviour** — fills, costs, the look-ahead test, the
  statistics, the live runner's replay equivalence, both golden references —
  is checked by the suite on a fresh clone, with no terminal and no network.
- **The measurements on real bars** — the spread-above-M1 finding, the
  broker's history depth, the quality report on 2025, the `tick_value` drift
  in the golden reference, the ambiguity share on the M1 baseline — were taken
  on one installation's `data_cache/` (fpmarkets, server Europe/Athens) and a
  clone cannot reproduce them without the same account and the same history.
  Each is labelled where it appears.

Run the tests, which do not need a terminal either:

```
.venv/Scripts/python -m pytest -q     # 587 pass, 9 skip on a fresh clone
```

Nine skips, and each says why. Eight are marked `mt5` and talk to the
MetaTrader 5 terminal. The ninth is the golden reference measured on real
XAUUSD.r bars, which only exist on the machine that downloaded them — but the
*second* golden reference, measured on the committed fixture, does run, so a
clone still has the guard that pins the engine's output to the trade.

Everything else — the engine, the cost model, the statistics, the API, the
live runner's replay equivalence — runs on generated or fixture data.

### With a real broker

To measure anything, you need real bars, and for those you need the terminal:

- Windows, with MetaTrader 5 installed, running and logged in
- `pip install MetaTrader5` (already in the `dev` extra on Windows)

```
.venv/Scripts/python -m examples.download_year --symbol EURUSD --year 2023
.venv/Scripts/python run.py
```

`run.py` prefers `data_cache/` over the fixture as soon as there is anything
in it, and says on start-up which of the two it is serving.

## Layout

```
core/data/provider.py      abstract DataProvider + SymbolSpec + Timeframe
core/data/mt5_provider.py  MetaTrader 5 implementation (read only)
core/data/servertime.py    server time <-> UTC conversions (pure, testable)
core/data/cache.py         Parquet cache per (symbol, timeframe, year) + JSON metadata
core/data/quality.py       quality report: gaps, duplicates, NaN, malformed bars
core/data/hc_reader.py     decoder for MT5's .hc cache (offline history)
core/data/fixture_provider.py  synthetic bars, so the repo runs without MT5
core/indicators/           pure functions + name -> function registry
core/indicators/incremental.py  bar-at-a-time state, bit-exact with the above
core/strategy/             pydantic spec, bar features, evaluator
core/strategy/incremental.py    the condition tree on one bar, no history walked
core/engine/               costs, sizing, risk gates, backtester
core/strategy/exits.py     exit distances in points, for the a-priori reports
core/metrics/              performance, break-even, buy & hold, uncertainty band, gates
core/research/edge.py      gate zero: does the signal beat the spread?
core/research/preview.py   what a spec will cost, before it is run
core/research/screen.py    screening funnel + the campaign's own trial count
core/research/manifest.py  a campaign's frozen inputs, so it can be re-run
core/runs/                 run store, orchestration, golden snapshots
core/validation/           walk-forward, permutation, multiple testing, tick resolve
core/batch/                same spec over many instruments + cross-sectional consistency
core/version.py            engine version (single source of truth)
api/                       FastAPI, 127.0.0.1 only
ui/                        Vite + React + TypeScript
strategies/                the JSON specs, including the library of classics
tests/                     pytest; integration tests are marked `mt5`
scripts/run_batch.py       batch runner driven by a YAML file
scripts/run_screen.py      screening campaign driven by a YAML file
scripts/verify_campaign.py re-runs a campaign from its manifest and diffs it
scripts/make_fixture.py    regenerates the committed synthetic dataset
scripts/run_live.py        the live runner, on closed bars
scripts/compare_live.py    the forward test's diary against a backtest of it
scripts/forward_test.ps1   start / stop / status / report, detached
scripts/migrate_spread_column.py  one-shot: rename the raw column, mark old runs
campaigns/                 the campaign this README's claim rests on, re-runnable
docs/forward-test.md       how to run the forward test and how to read it
docs/methodology.md        the choices that decide results, and why
docs/limitations.md        what this cannot tell you
fixtures/data_cache/       the committed synthetic dataset (invented data)
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

`risk.news_filter` must be `null` (it arrives with the live runner). A
declared but never referenced indicator produces a warning, not an error —
an ATR referenced only by an exit level counts as referenced.

### Exit levels

`stop_loss` and `take_profit` take three shapes, and which one is used
decides whether a spec means the same thing on two instruments:

```json
{"type": "points",  "value": 150}
{"type": "percent", "value": 0.15}
{"type": "atr", "indicator": "atr", "mult": 2.0}
```

150 points is 0.04% of price on gold and 0.14% on EURUSD: a point-based spec
run across ten instruments is a family of related strategies, not one
strategy ten times. The percent level is a share of the actual entry price;
the ATR level is a multiple of an ATR **read on the signal bar** — the last
closed bar when the trade was decided, since the execution bar's own range
does not exist yet — and frozen for the life of the trade. That is a
volatility-sized stop, not a trailing one: recomputing it bar by bar is a
different mechanism and is deliberately out of scope.

An ATR still in warm-up does not produce a fabricated distance: the entry is
skipped and counted under `exit_indicator_warmup`. Trades record the
`stop_level` and `target_level` they were actually given, because with a
variable distance those levels cannot be reconstructed from the spec
afterwards, and the tick resolution needs them.

Normalizing the exits does **not** normalize the cost of trading. Measured on
the ten instruments at M1: switching the baseline from 150/80 points to
2.0/1.07 ATR moved the spread from 6.3% of the stop distance to 21% of it on
average, because M1 volatility is small next to the spread on oil and silver.
The cross-instrument dispersion of mean R got wider, not narrower (std 0.066
→ 0.164). ATR sizing equalizes exposure to volatility; it does nothing about
the spread-to-volatility ratio, which varies just as much across instruments.

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

- **Spread**, bar by bar, from a column that is a spread. On M1 that is the
  feed's `spread`. **Above M1 the broker's field is the minimum of the M1
  spreads inside the bar** — measured at 100% on three instruments over four
  thousand periods each — so it is named `min_spread_m1` everywhere, and the
  cost model refuses to charge it under any name. A per-bar spread above M1
  is rebuilt from the M1 sample of the same period at the median or above
  (`core.data.spread.attach`); where that sample is missing the run is
  refused rather than served the column. Alternative policies `fixed` — the
  honest choice when the number comes from a measurement — and `quantile`
  for stress tests.

  What it was worth, June 2025, raw column against reconstruction: on
  GBPUSD.r, EURUSD.r and AUDUSD.r **daily** bars, 100%, 100% and 77% of fills
  were charged a spread of zero.
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

`runs/<run_id>/` with `spec.json`, `config.json`, `symbol_spec.json`,
`trades.parquet`, `equity.parquet`, `metrics.json`, `meta.json`. They open
with an editor and read with pandas.

`run_id` is **deterministic**: a hash of spec + configuration + data
fingerprint + the instrument's cost fields. Relaunching the same thing
neither recomputes nor duplicates; changing a parameter, the spread or a
single candle produces a different id, and the old run stays there
documenting how things were. The data fingerprint hashes the bar bytes, not
the dates: if the broker rewrites a candle, the time range does not notice —
the hash does.

### Why the SymbolSpec is part of the identity

The same batch, run twice, once gave XTIUSD -21.05 and once -19.86 under
identical spec and data hashes: the broker had changed its swap rates in
between, and nothing in the run recorded it. So each run now writes the full
instrument spec it used, with the timestamp it was read at, and `run_id`
covers the fields that decide what a trade costs — point, digits,
contract_size, tick_value, tick_size, swap_long, swap_short and the volume
limits. Name, currency and trade mode stay out: a broker relabeling does not
invalidate an id.

The drift is small and constant. Regenerating the golden reference between
two sessions moved the baseline's final equity from 89.60292671 to
89.60059376 with no engine change at all: `tick_value` had gone from
0.8628276588 to 0.8630212648, 0.02% of EUR/USD movement on an account in
EUR holding an instrument quoted in USD. Replayed against the pinned spec,
the engine reproduces the old figure exactly.

A run without `symbol_spec.json` — written before this existed — is reported
as "spec not registered", never assumed to have used the numbers on disk
today. Walk-forward, permutation, multiple testing and tick resolution all
revalidate a run against its own pinned spec rather than re-reading the live
one, or the drift would come back in through the side door.

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

### The uncertainty band, on every run

When one bar touches both stop and target the engine assumes the stop. On the
baseline that assumption was worth about twelve points of final equity over
eleven trades out of 134 — the difference between a run that loses 10% and
one that gains 1.5%. An assumption that decides the sign of the result is not
a footnote, so every run reports it as a band:

- **conservative** — every ambiguous trade exits on its stop. What the engine
  computed, and the lower bound.
- **optimistic** — every ambiguous trade exits on its target. The upper bound,
  and not a number to quote: it is there to size the room the assumption
  occupies.

Above a 5% share of ambiguous trades the run is declared not conclusive at
bar resolution, and the UI says so next to the equity rather than under it.

Gate zero estimates the same quantity **before** the backtest, from the
distance between the two levels and the distribution of bar ranges: among the
bars able to reach a level at all, what share could reach both. It is an
upper bound and a loose one — measured against realized runs it lands three
to five times above the actual share (23% against a realized 8.2% on the M1
baseline, ~7% against 0–1.3% on H1 and M15 cells) — so its warning threshold
is set on its own scale, at 20%. What it buys is dropping an untestable
configuration before spending a backtest on it.

### Tick resolution of ambiguous trades

`tick_resolve` pulls the ticks of the exit bar and asks which level came
first, reading a long on the bid and a short on the ask exactly as the engine
prices exits, then reports the delta against the conservative assumption. The
resolved figure always lands inside the band above: both use the same
accounting.

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
testing a family of related strategies, not one strategy ten times. ATR-based
exits fix that half of the problem and not the other half — see *Exit levels*.

## Screening a library of strategies

```
python -m scripts.run_screen campaign.yaml --json report.json --csv table.csv
```

A funnel in three stages of increasing cost: gate zero on every cell,
a backtest only where the gate passes, a permutation only on the survivors.
Spending a thousand resampled backtests on a signal with no directionality
produces a p-value about noise.

**The trial count is the point.** Three hundred backtests are three hundred
chances to be lucky: at the 5% level about fifteen come back "significant"
with no edge anywhere in the data. The campaign therefore counts every cell
it *started* — including those stopped at gate zero, which were attempts all
the same — and states what an observed Sharpe must reach before that count
stops explaining it. The variance of the Sharpe across trials is estimated on
the cells that reached a backtest and assumed to hold for the rest; that
assumption is printed with the result, as is the fact that cells sharing an
instrument or a strategy are not independent, which makes the correction
generous rather than strict.

### The campaign this README quotes

It is in [campaigns/](campaigns/), in two halves — the intraday grid over 2025
and the H1/H4/D1 grid back to 2020 — each with its YAML, its frozen manifest,
its JSON report and its table. The second carries the first's attempts and
results, so the correction it applies covers the whole search rather than the
half it happens to be running.

```
python -m scripts.verify_campaign campaigns/02-deep-history-2020.yaml \
    --manifest campaigns/02-deep-history-2020.manifest.json \
    --report   campaigns/02-deep-history-2020.report.json
```

Exit code 0 when every compared field of every cell matches. Both halves do.
Re-running needs the bars, which are not committed; `campaigns/README.md` says
what moved since the previous campaign and why.

`strategies/` holds ten classic rules — moving-average crossover, RSI and
Bollinger mean reversion, Bollinger and Donchian breakouts, MACD, stochastic,
ROC momentum, an EMA trend filter and a volatility-confirmed breakout — each
with the canonical parameters from its source, ATR exits so the files are
comparable across instruments, and, written in the file itself, whether it is
mean reversion or trend following and **which timeframe it was born on**.
Nine of the ten were designed for daily bars; running them on M5 is a
transplant, and the description says so rather than letting the table imply
otherwise.

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

## Building a strategy

A spec is JSON, and until phase 7 writing one meant writing JSON. The
**Strategy** page builds the condition tree with controls — add a condition,
pick an operator, pick operands, group into AND/OR, nest — over a vocabulary
served by `/api/vocabulary` straight from the indicator registry. Nothing in
the frontend duplicates that list, because a second copy is how a spec
becomes valid on screen and invalid on the server. The JSON sits alongside,
read-only, updating as you build: the format is learned by watching it
change.

What makes it worth having is the panel that says **what the strategy is
about to cost, before any backtest runs** (`core/research/preview.py`, one
API call, under a second on ten years of H4):

- the **signal frequency** over the chosen instrument and period, from
  evaluating the entry conditions alone — an upper bound on the trades, since
  a signal born while a position is open never becomes one;
- the **break-even win rate** the chosen exits imply, against the spread the
  instrument was measured at on M1;
- an a-priori estimate of the **share of ambiguous bars** — those wide enough
  to touch both the stop and the target, which bar resolution cannot settle;
- the **tradability verdict** for the pair, which is stage zero of the
  screening funnel;
- and, whether or not it is asked for, **how many attempts are already
  registered on this instrument and period, and what per-trade Sharpe a
  result would have to reach to survive that many.**

Under thirty expected trades it says so first and plainly: a result over
twenty trades is not a weak result, it is not a result.

The last point is the reason the panel is not optional. An editor makes
variants cheap to try, and trying variants is precisely how overfitting is
manufactured — so a run launched from the editor goes through `/api/backtest`
like every other, lands in the run store with its own spec hash, and is
counted by `collect_trials` along with everything else. There is deliberately
no path out of that page that produces a result the campaign does not count,
and `tests/test_api.py` holds a test that says so.

## The forward test

`rsi-mean-reversion` on `AUDUSD.r` H4 runs in dry run against the demo
account, writing every decision to a diary. It has no edge — it is the fourth
cell of a 530-attempt search at +0.1990 per trade against a required +0.6254,
and it was charged an assumed spread on every one of its 8,214 bars — and that
is not what is under test. The infrastructure is.

```powershell
./scripts/forward_test.ps1 -Start     # detached; closing the shell does not stop it
./scripts/forward_test.ps1 -Status
./scripts/forward_test.ps1 -Report    # the diary against a backtest of the same period
```

The report is a decomposition rather than a score: slippage on trades both
records took, the PnL of trades only one of them took with the reason the
diary gives, and **the rest** — which should be zero, and is the first line
to read. In dry run the slippage line should be zero too: no order was sent,
both sides are the same engine over the same bars, and a number there means
the runner and the backtester disagree.

Full instructions, including what it survives and how, in
[docs/forward-test.md](docs/forward-test.md).

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
- **Result** — the uncertainty band directly under the headline numbers
  (conservative, optimistic, what the gap is worth), equity and drawdown
  aligned on the same axis, metrics next to the buy & hold with the
  unsustainable-drawdown note, break-even win rate next to the realized one,
  the risk-gate table with a warning on any gate above 20% of the signals,
  the instrument specification the run used with its read timestamp, and
  paginated trades with the ambiguous ones highlighted.
- **Compare** — two or more runs overlaid, curves normalized to 100, metrics
  side by side with the delta against the first and the "better" direction
  declared per metric. The config fields that differ between the runs are
  shown under each column and as rows; the SymbolSpec fields that differ get
  their own flagged section, because those change what a trade costs.
  Identical configs are declared as such. Warns if the runs are on different
  instruments or periods, or if one predates SymbolSpec pinning.
- **Validation** — walk-forward (concatenated OOS curve, per-window table,
  IS/OOS degradation with standard errors, parameter stability), permutation
  with the null histogram and the real strategy marked on it, multiple
  testing with observed and deflated Sharpe side by side, and tick
  resolution of the ambiguous trades.
- **Batch** — the same spec across instruments, a sortable symbol x metric
  table with the standard error next to each estimate, and the
  cross-sectional consistency row in the lead position.
- **Screen** — a campaign over strategies x instruments x timeframes, run as
  a polled job whose id lives in the URL so a reload reattaches instead of
  losing twenty minutes of work. The trial-count panel sits above the table
  and stays there: a row with a good-looking equity is marked **not
  significant** unless its Sharpe clears the threshold corrected for the size
  of the campaign.

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
