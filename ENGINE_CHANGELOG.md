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
