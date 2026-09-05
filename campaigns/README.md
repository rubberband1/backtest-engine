# The campaign the README's claim rests on

Two halves of one search, 600 cells in all, over ten classic strategies and
ten instruments. Everything needed to re-run it and get the same numbers is in
this directory: the grid, the frozen inputs, the report and the table.

| | grid | period | cells | attempts |
|---|---|---|---|---|
| `01-intraday-2025` | 10 x 10 x M5,M15,H1 | 2025-01-01 → 2026-01-01 | 300 | 230 |
| `02-deep-history-2020` | 10 x 10 x H1,H4,D1 | 2020-01-01 → 2026-09-04 | 300 | 300 |

Cells and attempts differ because stage zero refuses a cell whose spread
exceeds 15% of one ATR before anything is tested on it. Seventy of the first
half's cells are refused that way, and a refusal is not an experiment: adding
it to N would make the correction look stricter while measuring nothing.

The second half carries the first half's attempts and results through
`prior_report`, so its panel corrects for **530 attempts** rather than
restarting at 300, and takes its maximum over the whole search rather than
over its own half.

## What it reports

- best cell of the search: `ma-crossover / XTIUSD / H1 / 2025-01-01..2026-01-01`,
  **+0.3307** Sharpe per trade over **32 trades**, on bars whose spread was
  measured on **74.9%** of the sample (4,417 of 5,897)
- a search of 530 attempts reaches **+0.3024** by luck alone
- to be credible at 95% confidence the observed value would have to reach
  **+0.6254**
- survivors after Bonferroni: **0**

## Re-running it

Each half writes its own manifest, and the manifest is what makes a re-run a
re-run rather than a second experiment: `tick_value` follows an FX rate and
the broker rewrites its swap table, so without frozen contracts every money
column moves between two executions for reasons that have nothing to do with
the code.

```
python -m scripts.run_screen campaigns/01-intraday-2025.yaml \
    --json campaigns/01-intraday-2025.report.json \
    --csv  campaigns/01-intraday-2025.table.csv \
    --manifest campaigns/01-intraday-2025.manifest.json

python -m scripts.run_screen campaigns/02-deep-history-2020.yaml \
    --json campaigns/02-deep-history-2020.report.json \
    --csv  campaigns/02-deep-history-2020.table.csv \
    --manifest campaigns/02-deep-history-2020.manifest.json
```

To check that the committed report still reproduces, instead of overwriting
it:

```
python -m scripts.verify_campaign campaigns/01-intraday-2025.yaml \
    --manifest campaigns/01-intraday-2025.manifest.json \
    --report   campaigns/01-intraday-2025.report.json
```

It exits 0 when every compared field of every cell matches, and prints the
cells that moved when they do not. It also prints how far the broker's
contracts have drifted since the manifest was frozen, because that is the size
of the difference the manifest is suppressing.

**This needs the bars.** `data_cache/` is not committed - six years of M1 on
ten instruments is not a repository - so a fresh clone can read these reports
but cannot reproduce them without downloading the history first
(`examples/download_year.py`). What a fresh clone *can* do is run the engine,
the tests and the dashboard on the synthetic fixture.

## Why the numbers moved from the previous campaign

The report these files replace was run before the manifest existed, under
engine 4.0.0. It is not in the repository, because a campaign nobody can
re-run is not evidence. These files were produced by engine 5.0.0, and the
differences from that earlier report, with their causes:

| | before | now |
|---|---|---|
| attempts | 600 | 530 |
| best observed | +0.1990 (172 trades) | +0.3307 (32 trades) |
| free at N | +0.3097 | +0.3024 |
| required | +0.4415 | +0.6254 |
| survivors | 0 | 0 |

- **600 → 530.** The first half was originally run by engine 3.2.0, which had
  no tradability stage, so all 300 of its cells were counted. Re-run today, 70
  of them are refused before testing. The grid is still 600 cells.
- **+0.1990 → +0.3307.** Two causes, and the larger one is a fixed defect: the
  panel used to take its maximum over the current campaign's cells while
  counting the attempts of both, so a better cell in the earlier half was
  invisible. `ma-crossover / XTIUSD / H1` scored +0.34477 in the original first
  half and was never reported. It now reads +0.3307 because the broker rewrote
  XTIUSD's swap.
- **+0.4415 → +0.6254.** N fell, which lowers the threshold; the winner's
  sample fell from 172 trades to 32, which raises it by more.
- **survivors: 0, both times.** The claim the README makes is unchanged.

Engine 4.2.0 and 5.0.0 both re-ran this campaign and reproduced these numbers
exactly: 4.2.0 is where the pooled maximum was fixed, and 5.0.0 changes only
the API path, which a campaign does not use.

### The broker rewrote a swap table mid-search

Between the previous campaign and this one, XTIUSD's swap went from
`+30.5 / -126.4` points to `-0.7 / -0.7`, and XBRUSD's from `38.0 / -156.4` to
`41.8 / -171.7`. Ten cells changed their trade count as a result, and **all ten
are on those two instruments**. The largest: `rsi-mean-reversion / XTIUSD / D1`
went from 21 trades and a busted account (final equity -15.18) to 38 trades and
+350.14, because at -126.4 points a night the account fell below the minimum
lot and stopped trading.

Every other difference is the `tick_value` drift of about 0.08%, which moves
money columns and leaves trade counts alone.

That is the entire argument for the manifest. Without one, a re-run mixes an
engine change with a broker change and no output can tell them apart.
