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
