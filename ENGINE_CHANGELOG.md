# Engine changelog

The engine version lives in `core/version.py` and follows semantic versioning
from the point of view of run results:

- **MAJOR** — results change for the same spec + config + data (execution
  rules, fill logic, cost accounting). Requires regenerating the golden
  reference with a changelog note.
- **MINOR** — new capabilities or metrics; existing results unchanged.
- **PATCH** — fixes with no observable change in any output.

The golden test (`tests/golden/`) pins the baseline strategy's complete trade
list and metrics on a frozen window (XAUUSD.r M1, 2025-04-02 through
2025-12-30 inclusive). Any unexplained difference fails the suite. To update
the reference after an intentional change:

```
python -m scripts.update_golden --update-golden --note "reason"
```

## 4.1.0

Phase 8: the repository made runnable without a broker, the live-versus-backtest
divergence closed with a proof, and campaigns made reproducible. **MINOR: no
result changes** — both golden references reproduce unchanged.

### The +0.0517 divergence was not the engine

`compare_live` reported +0.0517 on a dry-run replay diary, where the
difference can only be zero. It read as the simulator and the trader having
drifted apart — the failure this project exists to prevent — and the standing
hypothesis was that the diary predated the unification of execution in 4.0.0.

That hypothesis was wrong. Every decision in the diary matched to the bit:
entry and exit timestamps, prices, stop and target levels, lots, exit reasons,
bars held. Only the money columns differed, and all of them by one identical
factor, `1.000352809568884`, to within a single ULP across all five trades.
That factor is a ratio of `tick_value`: `0.8602076541277064` when the diary was
written against `0.8605111436193099` when it was compared. Re-running the
current engine with the older `tick_value` reproduces the old diary to one ULP.

The diary had not pinned the instrument spec it traded, so the comparison
re-read today's spec and rescaled one side of the diff. The residue landed on
the line labelled "slippage".

- the diary now pins the full `SymbolSpec` and its cost hash alongside the
  engine version, in the `started` event
- `compare_live` uses the diary's pinned spec, and reports a spec or version
  mismatch as `NOT COMPARABLE` instead of absorbing it into a total. It
  refuses outright to compare across engine versions without
  `--allow-version-mismatch`
- a test asserts a dry-run diary from the current engine compares to its own
  backtest at exactly `0.0` — not `approx`, which would have hidden this

### The repository runs without MetaTrader 5

- `core/data/fixture_provider.py`: a `DataProvider` over deterministic
  synthetic bars, with a real market week, a spread that moves with liquidity
  and every timeframe aggregated from one M1 path. The data is invented and
  says so at every level: `synthetic_fixture` in the cache provenance,
  `synthetic_fixture` in `/api/health`, and a banner on every page of the
  dashboard
- `fixtures/data_cache/`: 2.7 MB of that dataset, committed. `run.py` serves
  it whenever `data_cache/` is empty, which is the case on a fresh clone
- a second golden reference measured on the fixture, so the blocking
  regression guard runs on a machine that has never seen a broker. Previously
  it skipped there, and the suite passed while the one test that pins the
  engine's output never executed

### Campaigns are reproducible

- `core/research/manifest.py`: a campaign freezes every instrument spec, the
  server clock and every setting that changes a number
- `scripts/verify_campaign.py`: re-runs a campaign from its manifest and diffs
  it cell by cell. Demonstrated with `tick_value` moved 0.13% underneath it —
  without the manifest the net PnL moved (−47.53 → −47.59); from the manifest
  every compared field matched and the drift was reported as drift
- the campaign report declares its manifest's freeze date and the frozen cost
  fields

### The spread says whether it was measured or assumed

- `core.data.spread.coverage`: the share of a run's bars that had an M1 sample
  behind them. A bar inside a hole in the M1 sample counts as assumed, not as
  measured
- reported on every run, every campaign cell, and aggregated in the campaign
  panel's assumptions. Measured on the current cache: AUDUSD.r H4 **0.0%**,
  XAUUSD.r H1 11.5%, EURUSD.r H1 2.4% — the first of those is the campaign's
  own best cell

### Interface and tooling

- numeric inputs no longer follow the browser's locale: on an Italian machine
  the editor showed `0,4` beside a JSON panel reading `0.4`. A `NumberField`
  parses and renders with a dot, and accepts a typed comma
- the editor states which multiple-testing scope each threshold belongs to and
  shows both: the whole search and this instrument alone
- the Spec panel wraps long values instead of truncating them
- `vite.config.ts` reads the API port from the environment; `run.py --api-port`
  previously started a backend the dashboard could not reach
- ruff and mypy over `core/`, both clean, in CI on Linux and Windows for
  Python 3.10 and 3.12. mypy found a real defect: `FixtureProvider.probe_depth`
  constructed a class that does not exist

## 4.0.0

Phase 7: the three declared debts closed, a strategy builder, and a forward
test. **MAJOR because two of the debts change results**: the spread charged
above M1 and the bar on which a time stop fires.

### Changed - results move

- **The aggregated spread column can no longer become a cost.** Phase 6
  measured that the `spread` field of a bar above M1 is the *minimum* of the
  spreads of the M1 bars inside it, and fixed the one place that read it.
  A fix in one place is not a guarantee, so the mistake is now
  unrepresentable rather than merely absent:
  - above M1 the raw field is called **`min_spread_m1`** everywhere - in the
    cache files, in every frame the engine hands around, in the spec
    vocabulary. `ParquetCache.read_year` renames on the way out, so no
    reader can reach it by writing `bars["spread"]` whatever the file says;
  - `spread_mode="per_bar"` (and `"quantile"`) above M1 requires a `spread`
    column **rebuilt from the M1 bars of the same period**, at the median or
    a higher quantile, never the minimum. `core.data.spread.attach` builds
    it, assigning each M1 bar to the bar that contains it rather than to a
    resample bin - this broker stamps H4 bars at 21:00 server time and an
    epoch-aligned grid lines up with none of them;
  - where the M1 sample does not cover the period, the run is **refused**
    (`SpreadUnavailable`, HTTP 409) instead of falling back to the column.
    Charging a per-bar spread that cannot be measured is exactly the failure
    being closed, so it is not offered as a degraded mode;
  - `per_bar_spread_quantile` (default 0.5, floor 0.5) is part of `RunConfig`
    and therefore of the `run_id`: the reconstruction changes every cost in
    the run, and two different reconstructions are two different runs.

  What it was worth, measured on June 2025, raw column against
  reconstruction:

  | instrument | tf | raw median | bars charged zero | rebuilt median | rebuilt mean |
  |---|---|---|---|---|---|
  | GBPUSD.r | H1 | 0.0 | 63.9% | 4.0 | 6.42 |
  | GBPUSD.r | D1 | 0.0 | 100.0% | 3.0 | 3.09 |
  | EURUSD.r | H1 | 0.0 | 53.6% | 1.0 | 1.89 |
  | EURUSD.r | D1 | 0.0 | 100.0% | 1.0 | 1.00 |
  | AUDUSD.r | D1 | 0.0 | 77.3% | 3.0 | 2.91 |
  | XTIUSD | H1 | 33.0 | 0.0% | 40.0 | 52.99 |
  | XAUUSD.r | H1 | 7.0 | 0.0% | 7.5 | 8.54 |

  On the FX instruments' daily bars every single fill was free.

- **The time stop counts session bars on the server clock.** A broker
  session is fixed in server local time; in UTC that same hour moves twice a
  year, so a weekly slot grid laid out in UTC has slots that exist for part
  of the year and not the rest, and the threshold keeps one regime and calls
  the other a hole. `SessionCalendar` now lays its grid in UTC - which has
  every instant exactly once - and reads each instant's weekly slot **on the
  session clock**. The calendar carries the clock it was inferred on, so a
  pinned one still counts the way it did. `periods_per_year` takes the same
  correction, which reaches the annualized Sharpe, Sortino, annual return
  and Calmar.

  Measured impact, with the clock as the only thing that changed:
  - **on the baseline: none.** The golden window was replayed with the
    pinned SymbolSpec: 0 differences against the 3.2.0 reference. The
    baseline has no time-stop exits, and its inferred slot count (6893 per
    week) is the same on both clocks.
  - **on the campaign: real, and not negligible.** Every one of the 88
    phase-6 cells that reached a backtest was run twice over the same bars,
    the same spec and the same costs, differing only in the clock the
    session calendar was inferred on. 20 have at least one time-stop exit
    and **7 move**. The largest is roc-momentum on AUDUSD.r D1: per-trade
    Sharpe from -0.2154 to -0.0807 - 0.135, against a free-Sharpe allowance
    of 0.315 at 600 attempts - and final equity from -10.49 to +54.02, on 33
    trades becoming 35. Cells moving by more than a third of the allowance
    is why the campaign was re-run rather than patched.

  The re-run, under 4.0.0 and including the spread change above:

  | | phase 6 | phase 7 |
  |---|---|---|
  | attempts | 600 | 600 |
  | cells backtested | 88 | 88 |
  | variance across trials | 0.010277 | 0.009938 |
  | free Sharpe at N=600 | +0.3150 | +0.3097 |
  | required Sharpe/trade | +0.4469 | +0.4415 |
  | best observed | rsi-mean-reversion / AUDUSD.r / H4, +0.19897 (172 trades) | the same cell, the same +0.19897 |
  | survivors | 0 | 0 |

  Ten cells differ between the two reports rather than seven, and the extra
  three are **not** the engine: `tick_value` on those instruments moved
  between the two runs (on EURUSD.r, 0.8601928552 to 0.8612670961), which
  changes money-per-point on every trade and the lot size once equity
  crosses a step. Two campaigns run months apart are not a controlled
  comparison of anything, which is what the isolated measurement above
  exists for. The conclusion is unchanged and was never in doubt: nothing
  clears the threshold, and the best cell is short of it by more than a
  factor of two.

### Changed - results do not move

- **The incremental indicator state.** The live runner recomputed every
  indicator over its whole retained history on every bar, which made a
  replay quadratic in the number of bars. Indicators now expose a state
  updated one bar at a time, replicating pandas' own arithmetic
  (`add_mean`/`remove_mean` with Kahan compensation, Welford for the
  variance, the `adjust=False` EWM recursion including its division by
  `old_wt + new_wt`, which is not 1.0) rather than approximating it.

  The obvious alternative was measured first and rejected: recomputing over
  a trailing window is **never** exact for a rolling mean, because pandas'
  compensation term depends on every value that has ever entered the window.
  An `sma(20)` recomputed over the last 20, 100, 500, 2000 or 5000 bars
  misses the full-series value on a third of the bars at every length;
  `bollinger` and `stoch` miss on all of them.

  `tests/test_incremental.py` demands bit equality with a from-scratch
  recomputation over 5000 bars, on all nine indicators and eighteen
  parameter sets, including flat stretches, price jumps and exactly repeated
  values. **All nine are exact; `NOT_EXACT` is empty.** A tenth indicator
  cannot be registered without landing in one list or the other.

  `SessionCalendar` caches its expected-bar grid and extends it by doubling
  for the same reason: computing one ordinal rebuilt the grid from the
  origin, which was the other half of the quadratic. Replaying AUDUSD.r H1
  now costs 0.37s / 0.66s / 1.32s / 2.64s over 1000 / 2000 / 4000 / 8000
  bars - linear - against 3.6s / 7.4s / 15.6s / 34.6s recomputing, and the
  trades are identical on every one of them. The whole test suite got 20%
  faster as a side effect.

- **A stale quote can no longer name the server timezone.** The offset is
  measured by comparing a live tick's wall clock against the local one and
  rounding to the nearest half hour. With the market closed the newest tick
  is as old as the session has been shut, and the rounding hides that age:
  measured here on a Friday evening, fifty minutes after the close, the last
  XAUUSD.r tick made an Athens server (UTC+3) read as 2h09 and resolve to
  **Europe/Berlin** - a real timezone, an hour from the right one, in the
  value the session calendar, the swap accounting and every historical
  conversion are built on. A measurement further than two minutes from a
  half-hour multiple is now refused, and the caller falls back to the value
  the cache recorded during a session, as it was already written to do.

  The `mt5`-marked tests skip on that condition instead of failing: they
  read bars and ticks, both expressed on a clock that cannot be measured
  with the market closed.

- **`scripts/update_golden.py` replays the pinned SymbolSpec.** It read the
  instrument spec from the terminal, so every regeneration was also a silent
  data change: the first 4.0.0 attempt produced 548 differences, every one
  of them a `tick_value` that had drifted from 0.8630 to 0.8612 since the
  reference was written, and none of them anything the engine had done.
  `--refresh-symbol-spec` asks for that rebase deliberately, and the
  changelog entry says which was used.

### Added

- **`core/indicators/incremental.py`** - bar-at-a-time state for the nine
  indicators, plus `core/strategy/incremental.py`, which evaluates the
  condition tree on scalars so the live runner touches one bar per bar.
- **The Strategy page** - a visual builder for the condition tree, the
  exits, the sizing and the risk gates, validated against the API as it is
  typed, with each message shown under the control that caused it and the
  JSON alongside, read-only. Import a spec, duplicate one of the library's
  ten, save to `strategies/` with a refusal rather than an accidental
  overwrite.

  `GET /api/vocabulary` serves what a spec may contain - indicators, their
  parameters with defaults and bounds, outputs, features, bar fields,
  operators, exit types - straight from the registry, so the editor cannot
  drift from what the engine accepts.

  `POST /api/strategies/preview` (`core/research/preview.py`) reports what a
  spec is about to cost before any backtest: signal frequency, the
  break-even win rate the exits imply against the instrument's M1-measured
  spread, an a-priori ambiguous share, the tradability verdict, and how many
  attempts are already registered on that symbol and period with the
  per-trade Sharpe they require. Under thirty expected trades it says so
  first. On ten years of AUDUSD.r H4 it answers in 0.29s.

  Runs launched from the editor go through `/api/backtest` like any other,
  including specs that have never been saved: they land in the run store
  with their own spec hash and are counted by `collect_trials`. A test
  asserts that there is no path out of the page that escapes the count.
- **`scripts/migrate_spread_column.py`** - renames the raw column in the
  cache files and marks every stored run that charged it
  (`aggregated_spread_cost`), which the API serves on the run detail and the
  listing: those numbers are not comparable with anything produced since.
  Applied here to 278 above-M1 files and 360 runs.

- **The forward test.** `rsi-mean-reversion` on `AUDUSD.r` H4, dry run,
  demo account, started by `scripts/forward_test.ps1` through
  `Win32_Process.Create` rather than `Start-Process` - a child started the
  ordinary way joins the caller's Windows job object and is killed with it,
  which was not a theory: started from inside a harness task, the runner was
  gone the moment that task was reaped, seconds after `-Status` reported it
  running. Spawned through WMI it belongs to no such job, and it writes its
  own log (`--log-file`) so no shell wrapper has to survive alongside it.
  Built to be left alone: a failed fetch backs off from 20s to a five-minute ceiling
  instead of dying, a restart reads the open position back from the broker
  by magic number and the closed trades and equity back from the diary, and
  the session calendar carries the server clock so a DST change moves
  neither the session nor the bar a time stop fires on.
  `scripts/compare_live.py` diffs the diary against a backtest of exactly
  the period it covers, decomposed into slippage, trades only one side took,
  and the remainder - and says plainly that in a dry run the first of those
  should be zero too, because both sides are the same engine and a number
  there means they disagree. `docs/forward-test.md` says how to run it and
  how to read it.

---

## 3.3.0

> **Reconstructed.** This section was written during phase 6 and never
> committed; it was lost from the working tree during phase 7 by a
> `git checkout` that reverted more than it was meant to. The text below is
> the original where it survived in the phase-7 session record, and rewritten
> from the phase-6 code where it did not. The facts and figures are
> re-derived from `core/data/spread.py`, `core/live/`,
> `core/research/tradability.py` and the tests; the prose is not
> word-for-word the original.

Phase 6: deep history, a tradability gate before the funnel, and a live runner
that shares the backtester's execution engine.

**No execution rule, fill or cost formula changed.** The bar loop was lifted
out of `backtester.py` into `core/engine/execution.py` unchanged, and the
golden reference matches to the last decimal - which is the point: the live
runner drives that same state machine, so "the backtest and the live system
are the same system" is a property of the code rather than a claim about it.

### Found

- **The `spread` column of a bar above M1 is the minimum spread inside the
  bar, not a spread.** Verified against the M1 sub-bars on XAUUSD.r, GBPUSD.r
  and XTIUSD: it equals the sub-bar minimum in **100.0%** of more than four
  thousand hours each. Charging it as a fill cost charges the best price of
  the period. The size of the error, M1 median against the H1 column:
  XTIUSD 29 -> 4 points, GBPUSD.r 3 -> 0, USDJPY.r 3 -> 0, XAUUSD.r 7 -> 2.
  On four FX instruments the median H1 spread is exactly zero and 67-76% of
  their H1 bars carry a zero spread. Every backtest above M1 with
  `spread_mode="per_bar"` has therefore been trading for free on most bars.
  Phase 5's campaign ran that way; its costs were understated, and since it
  still found nothing its negative conclusion survives - but any positive
  result it had produced would have been an artefact.
- **The deflated-Sharpe variance was contaminated by tiny samples.** A
  per-trade Sharpe over two trades is a ratio, not an estimate: cells with
  2-7 trades reached -10.1, and including them took the variance across
  trials from 0.011 to 2.08 and the corrected threshold from a reachable
  +0.60 to an unreachable +4.77. A threshold nothing can clear is not a
  strict test.
- **The quality report inferred the session calendar in UTC**, so a DST
  change moved every weekly slot by an hour and the whole summer read as
  missing. On EURUSD.r D1 over 27 years that reported 40.8% completeness for
  a series with no real holes; on the server clock the same series reads
  99.6%.
- `core/data/servertime.py` caught `pd.errors.AmbiguousTimeError`, which does
  not exist: the `except` clause itself raised `AttributeError`, so the
  intended fallback had never run. It surfaced the first time history deep
  enough to contain an unresolvable DST transition was loaded.

### Added

- **`core/engine/execution.py`** - the execution rules as one bar-at-a-time
  state machine (`Executor`, `BarInput`). `backtester.py` is now a vectorized
  driver over it; the live runner is a streaming driver over the same object.
- **`core/live/`** - `runner.py` (closed bars only, pinned session calendar,
  start-up reconciliation by magic number, PID lock), `broker.py` (order
  sending, retcodes by name, partial fills, and a guard that refuses a
  non-demo account when `--send` is given), `journal.py` (append-only JSON
  diary, fsynced per line, with account identifiers scrubbed at any nesting
  depth), `lock.py`, `replay.py` and `compare.py`.
- **`tests/test_replay_equivalence.py`** - blocking. The runner replays
  history one bar at a time and its trades must equal the backtester's
  exactly: same entry and exit timestamps, prices, levels, lots and exit
  reasons, with no tolerance. Covers the baseline plus four library
  strategies, session gaps, a bar-by-bar spread, commission and swap, ATR
  exits, the gate accounting and the equity curve.
- **`core/research/tradability.py`** - stage zero of the funnel. Median
  spread over median ATR per instrument x timeframe, refused above 15% of one
  ATR, with the spread measured on M1 and never on the bar being judged. A
  refused cell is not a failed experiment: it never counted as an attempt,
  so it does not inflate the multiple-testing correction.
- **`core/data/spread.py`** - the spread distribution measured where it means
  something (M1, per instrument), stored next to the cache, plus
  `check_aggregation`, which re-checks the claim above on demand.
- **`core/data/depth.py`** and **`core/data/continuity.py`** - how far back
  each instrument really goes, and whether two sources agree where they
  overlap. Deep history can arrive from the live feed or be decoded out of
  the terminal's `.hc` cache; stitching them is only legitimate if they say
  the same thing, so the seam is measured rather than averaged away.
- **Cache schema 2** - per-interval provenance. Each stretch of coverage
  records which source supplied it, so "my backtest used broker data" stops
  being an unverifiable claim. Version 1 files still load, and their coverage
  is reported as provenance `unrecorded`.
- **`scripts/extend_history.py`**, **`scripts/run_live.py`**, the **Live**
  page, and history extended to 2020 on H1/H4/D1 for ten instruments.

### Fixed

- The deflated Sharpe now excludes cells below the minimum trade count from
  the variance across trials, and says how many it excluded.
- `core/data/quality.py` infers the session calendar on the server clock when
  one is passed, and the report states which clock it used.
- `core/data/servertime.py` catches the exception that actually exists.

---

## 3.2.0

Phase 5A: cross-instrument correctness. No execution rule, fill or cost
formula changed - verified before regenerating the golden reference by
replaying the frozen window with the *pinned* SymbolSpec: 268 differences
against 3.1.0, every one of them a newly added field, zero changes to any
computed value.

### Added

- **SymbolSpec pinning** (`core/data/provider.py`, `core/runs/store.py`).
  Every run persists `symbol_spec.json` with the full instrument spec and the
  timestamp it was read at, and `run_id` now hashes the fields that decide
  what a trade costs: point, digits, contract_size, tick_value, tick_size,
  swap_long, swap_short, volume_min/max/step. Descriptive fields (name,
  currency, trade mode) stay out, so a broker relabeling does not invalidate
  every id. **This invalidates every run_id created before 3.2.0**, which is
  the intended effect: the same batch run twice gave XTIUSD -21.05 and -19.86
  under one id because the broker had changed its swap rates in between.
  A run without `symbol_spec.json` is reported as "spec not registered" and
  never assumed to have used the spec on disk today. Walk-forward,
  permutation, multiple-testing and tick-resolve now revalidate a run against
  its *pinned* spec instead of re-reading the live one.
- **Normalized exits** (`core/strategy/spec.py`, `core/strategy/exits.py`).
  `stop_loss`/`take_profit` accept `{"type":"points"}`, `{"type":"percent"}`
  (a share of the entry price) and `{"type":"atr","indicator":...,"mult":...}`
  (a multiple of an ATR read on the signal bar, frozen for the trade - a
  volatility-sized stop, not a trailing one). An ATR still in warm-up skips
  the entry (`exit_indicator_warmup`) rather than inventing a distance.
  Trades now record `stop_level` and `target_level`: with variable exits the
  distance cannot be reconstructed from the spec afterwards, and tick
  resolution reads them off the record.
- **Uncertainty band on every run** (`core/metrics/ambiguity.py`). Ambiguous
  trades, their share, and what they are worth in equity between the two
  extreme readings (all stops vs all targets). Above a 5% share the run is
  declared non-conclusive at bar resolution. Gate zero estimates the same
  quantity a priori, from the distance between the levels and the
  distribution of bar ranges, so an untestable configuration can be dropped
  before it is run.
- **Gate accounting** (`core/metrics/gates.py`). Risk decisions carry a
  stable `code`, so rejections aggregate per gate instead of per message -
  counting on the human reason produced one bucket per spread value, which is
  how a `max_spread_points` gate rejecting half the signals stayed invisible.
  Signals arriving while a position is already open are counted too, instead
  of vanishing before the gates. A gate above 20% raises a warning.
- Break-even win rate over variable exits is reported as a weighted average
  with the dispersion of both sides attached, instead of being refused as
  "dispersed amounts". `strategies/rsi-wick-atr.json`: the baseline signal
  with SL 2.0 ATR / TP 1.07 ATR, the same 1.875 risk/reward as 150/80 points.

### Fixed

- `.gitignore` matched `core/runs/` with the pattern meant for the persisted
  `runs/` output directory: four source files of the engine had never been
  committed. Both patterns are now anchored to the repo root.

### Golden reference

Regenerated, and the final equity moves from 89.60292671 to 89.60059376 for a
reason that has nothing to do with the engine: `tick_value` was 0.8628276588
when the reference was pinned and 0.8630212648 when it was regenerated, +0.02%
of FX drift on an account in EUR against an instrument quoted in USD. Replayed
against the old pinned spec the engine reproduces the old number exactly. This
is the drift A1 exists to make visible, caught here on the one run where
everything else was frozen.

## 3.1.0

Phase 4: statistical validation and batch execution. No execution rule, cost
or fill changed, so the golden reference is untouched and every run made at
3.0.0 reproduces identically. Only the engine version inside `run_id`
changes, which is what a MINOR bump is for: new runs get new ids, old runs
stay readable.

### Added

- `core/validation/walkforward.py` — rolling or anchored windows, an optional
  parameter grid optimized in-sample and applied out-of-sample, an embargo
  between the two legs as wide as the longest holding the spec can produce,
  and a minimum in-sample trade count (default 30) below which a window is
  discarded and reported as discarded. Output: the concatenated OOS curve
  (the only series readable as performance), a per-window table, the IS→OOS
  degradation with standard errors, and the stability of the chosen
  parameters across windows.
- `core/validation/permutation.py` — two null models, both executed through
  the real `Backtester`: **random entries** (same exit rules, costs, gates and
  long/short signal counts, random timestamps) and **permuted returns**
  (block bootstrap of the price path, bar shape preserved as log offsets from
  each bar's close). Empirical p-value in the conservative form
  (1 + #{null ≥ observed}) / (1 + N), which can never reach zero.
- `core/validation/multiple_testing.py` — Deflated Sharpe Ratio (Bailey &
  López de Prado) on the *per-trade* Sharpe, corrected for the trial count,
  skew and kurtosis; Bonferroni and Benjamini-Hochberg thresholds reported
  side by side; PBO via CSCV when a grid is supplied, and an explicit
  statement that it was not computed when one is not. Trials are counted from
  the run store by distinct spec hash over the same data, never typed in.
- `core/validation/tick_resolve.py` — resolves ambiguous trades (stop and
  target touched in the same bar) against tick data, reading a long on the bid
  and a short on the ask exactly as the engine prices exits. Missing ticks,
  a closed terminal or a period older than the broker's tick history are all
  reported as unresolved; nothing is ever assumed in their place.
- `core/batch/runner.py` and `scripts/run_batch.py` — the same spec across
  many instruments, periods and grid cells in parallel, every cell a normal
  run in the store. The output that matters is the cross-sectional
  consistency block: sign agreement across instruments with a binomial sign
  test, the spread of the metric, and a flag when the aggregate is carried by
  a single instrument.
- API: `POST /api/validation/walkforward`, `/permutation`,
  `/multiple-testing`, `/tick-resolve` and `POST /api/batch`.
- UI: `Validation` and `Batch` pages.

### Changed

- `Backtester.run` accepts an optional `signals` argument that replaces the
  ones the spec would generate. It exists so the permutation nulls go through
  the same fills, costs and gates as the strategy they are compared against.
  Default behaviour is unchanged and the golden test confirms it.
- The UI is entirely in English, matching the rest of the repository.
- `PyYAML` is now a declared dependency (the batch CLI reads a YAML file).

### Known limits, stated rather than hidden

- 1000 permutation iterations over nine months of M1 take about 130 s
  (random entries) and 150 s (permuted returns) on this machine, above the
  120 s target. The worker pool is capped at 8 processes because each one
  holds its own copy of the bar series; raising the cap on a machine with more
  free memory brings it under two minutes.
- PBO is only computed when a grid is passed. Without candidates to compare,
  CSCV has nothing to cross-validate, and the report says so instead of
  returning a number.

## 3.0.0

Initial versioned release (phases 1-3: data layer, event-driven backtester,
research gate, run store, API, UI).

### Post-mortem: the phase 2 vs phase 3 discrepancy

Between the end of phase 2 and the end of phase 3 the reported baseline
results changed with no declared engine change:

| | trades | final equity | spread paid | gross PnL |
|---|---|---|---|---|
| phase 2 | 135 | 88.30 | 8.70 | -3.01 |
| phase 3 | 134 | 89.59 | 8.64 | -1.77 |

Investigation (2026-09-02): **the engine did not change.** The current engine
reproduces both rows exactly from the same code:

- the full cached dataset (265,609 M1 bars, through 2025-12-31 21:59 UTC)
  yields 135 trades, final equity 88.31, spread 8.69, gross -3.00 — the
  phase 2 numbers;
- the phase 3 run was launched with `end = 2025-12-31T00:00Z`, which the
  exclusive end bound (`bars.index < end`) cuts to 264,289 bars ending
  2025-12-30 23:59 — 134 trades, final equity 89.59, spread 8.64,
  gross -1.77.

The missing trade is the last one of the period: a short entered
2025-12-31 06:57 UTC that hit its stop loss (net -1.29, exactly the delta
between the two equities). "Same period" was not the same period: the phase 3
run dropped December 31. The golden test exists so that this class of silent
drift — data window, data content, or engine behavior — fails loudly instead.

A second, smaller source of drift surfaced during the investigation: the
symbol spec cached on disk is refreshed from the terminal whenever the
resolver reaches it, and its tick_value (account currency) moves with the
EUR/USD rate. Two otherwise identical runs hours apart differ by ~0.1% in
every money amount for that reason alone. The golden reference therefore pins
the SymbolSpec and the server timezone inside the reference file and replays
them instead of asking the terminal.

### Golden reference update (2026-09-02, engine 3.0.0)

- trades: 134, final equity: 89.60292671
- reason: Initial golden reference: baseline frozen at engine 3.0.0 after the phase 2/3 window discrepancy was traced to a data-window difference (Dec 31 excluded), not an engine change.

### Golden reference update (2026-09-02, engine 3.0.0)

- trades: 134, final equity: 89.60292671
- reason: Pin SymbolSpec and server timezone inside the reference: tick_value moves with FX rates and must not make the golden drift.

### Golden reference update (2026-09-02, engine 3.0.0)

- trades: 134, final equity: 89.60292671
- reason: Baseline spec description translated to English (repo-wide language migration). Signals, trades and metrics are unchanged; only the spec hash moved.

### Golden reference update (2026-09-02, engine 3.2.0)

- trades: 134, final equity: 89.60059376
- reason: Phase 5A: trades now record the stop and target levels they were actually placed at (ATR and percent exits cannot be reconstructed from the spec afterwards). Verified before regenerating: 268 differences against the 3.1.0 reference, all of them the two new fields, zero changes to any computed value.

### Golden reference update (2026-09-04, engine 4.0.0)

- trades: 134, final equity: 89.60059376
- instrument spec: replayed from the existing reference
- reason: Phase 7 A2: session bars are counted on the server clock instead of UTC, so a DST change no longer shifts the weekly slot grid the time stop counts on. The baseline does not move: 134 trades, final equity 89.60059376, every trade and every metric identical to the 3.2.0 reference, because none of its trades exits on the time stop and the inferred slot count per week (6893) is the same on both clocks. Regenerated to record that the check was made and came back empty, and to stamp engine 4.0.0.

### Golden reference update (2026-09-05, engine 4.0.0)

- fixture: 263 trades, final equity 155.38121167 (read from the terminal (the reference is rebased onto them))
- reason: Add a golden reference on the synthetic fixture, so the engine's blocking regression guard runs on a clone with no MT5 terminal and no downloaded data.
