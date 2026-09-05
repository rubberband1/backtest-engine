# Limitations

What this engine cannot tell you, and what would be wrong to conclude from it.
None of this is hypothetical caution; each item below has changed or bounded a
result that was actually produced.

---

## One broker, one account, one feed

Every bar came from a single MetaTrader 5 broker (FP Markets), through one
account, on one server clock (Europe/Athens). Nothing here is cross-checked
against a second source.

That matters more than it sounds. The spread, the swap rates, the tick value,
the exact timestamps of bars and the instrument specification are all *this
broker's*. A strategy that clears the cost hurdle here might not clear a
wider-spread broker's, and one that fails here might pass elsewhere. Where a
result depends on the broker's cut - which, for everything short-horizon,
is most of them - it is a result about trading **at this broker**.

There is also no verification that the broker's own history is correct. A
rewritten candle is detectable (a run stores a fingerprint of the bar bytes,
so the same date range with different data is a different run) but a candle
that was always wrong is not.

---

## The history starts in 2020, and the older part is thinner

Depth by timeframe, as probed from this broker:

- **H1/H4/D1**: back to 2020-02 for metals and oil, 2021-05 for FX majors.
- **M1**: only recent - a few months, not years, and not uniformly.
- **Ticks**: not reliably available at all (see below).

So the campaign's older bars are the ones with the least corroboration. Two
consequences:

- Anything that needs M1 - the honest per-bar spread most of all - cannot be
  reconstructed for the older part of the sample.
- EURUSD's pre-2020 bars are backfill rather than this broker's own record,
  and are not used.

Six years is not a long sample for a claim about market structure. It contains
one pandemic, one inflation cycle and one rate-hiking cycle, and no 2008.

---

## The historical spread is assumed, not measured - and the assumption's
## direction was checked

This is the largest open weakness in the cost model, and it is worth being
precise about which way it cuts.

**What the engine does.** Above M1 the broker's `spread` field is the
*minimum* of the M1 spreads inside the bar and is refused as a fill cost (see
`methodology.md` §2). Where M1 exists, the spread is reconstructed from it.
Where it does not - which is most of the sample before 2025 - a **fixed spread
measured on recent M1 is charged to bars that are years older**: 3.0 points
for AUDUSD.r, 7.0 for XAUUSD.r.

That is exactly the class of error the `min_spread_m1` correction was about:
a cost stated with more confidence than the data supports, on the oldest part
of the sample, which is the part added to buy statistical power.

**Can it be measured instead?** Only patchily. Probing this broker's tick
history one day at a time:

| year | AUDUSD.r | XAUUSD.r |
|---|---|---|
| 2021 | no ticks on any sampled day | no ticks on any sampled day |
| 2022 | none | 1 of 4 days |
| 2023 | 3 of 4 days | 4 of 4 days |
| 2024 | none | none |
| 2025 | 2 of 4 days | 2 of 4 days |
| 2026 | 2 of 4 days | 2 of 4 days |

The gaps are not a server-side history boundary so much as MetaTrader's local
tick cache: the terminal holds the ticks it has been asked for. Either way,
**there is no tick series deep and continuous enough to measure the spread
across the 2020-2023 sample**, so the fixed assumption cannot simply be
replaced with a measurement.

**Which way is the assumption wrong?** The expectation was that spreads years
ago were *wider*, so that charging today's number understates costs on the old
bars and flatters the result. On the days where ticks exist, that is
backwards. Median tick spread, in points:

| year | AUDUSD.r | XAUUSD.r |
|---|---|---|
| 2022 | — | 7.0 |
| 2023 | **2.0** | 8.5 |
| 2025 | 3.0 | 9.5 |
| 2026 | 6.5 | 11.0 |

Spreads have **widened** over the period, not narrowed. Charging AUDUSD's
recent 3.0 points to 2023 bars where the measured median was 2.0 makes the
backtest *more* pessimistic than reality, not less; on gold the assumed 7.0
against a measured 8.5-9.5 goes the other way, but by a smaller proportion.

**Do not over-read this.** It is three or four sampled days per year, on two
instruments, from a cache with large holes. It establishes the direction, not
the magnitude, and it says nothing at all about 2021 or 2024, where no ticks
exist. The correct summary is: *the historical spread is an assumption, its
error is probably conservative on AUDUSD, and it remains the input most
capable of moving a marginal result.*

**How much of the sample is in that position.** Now measured, and reported on
every run, every campaign cell and the campaign panel:

| instrument / timeframe | bars with an M1 sample behind them |
|---|---|
| AUDUSD.r H4 | **0.0%** (0 of 8,214) |
| XAUUSD.r H1 | 11.5% (4,436 of 38,510) |
| EURUSD.r H1 | 2.4% (4,167 of 171,813) |

The first row is the campaign's own best cell. **The best result of the
600-attempt search - rsi-mean-reversion on AUDUSD.r H4, +0.19897 per trade
over 172 trades - was charged an assumed spread on every single bar**,
3.0 points measured over 2025-05-01 to 2025-12-31 and applied back to 2021.

Taken with the tick measurement above, the direction is favourable: AUDUSD's
2023 median was 2.0 points against the 3.0 charged, so that cell was, if
anything, billed too much. But "the assumption happens to be conservative"
is a different statement from "the cost was measured", and only the second
one is a result. The cell does not clear its threshold either way.

---

## Two different trial counts are reported, and they answer different questions

The Result panel reports the correction for the attempts made **on that
instrument and period**; the campaign report corrects for the **entire
search**. Both are defensible and they are not the same number - 50 attempts
and 600 attempts give different thresholds for the same observed value.

A reader who sees one figure in one place and a different one in another,
without being told which scope each belongs to, will reasonably conclude one
of them is wrong. Both are now labelled and both are shown: the editor's KPI
leads with the whole-search threshold - the one that governs a claim of
discovery - and names the per-instrument one underneath, with a table giving
the attempts, the expected-by-luck value and the required value at each
scope. On the current store that reads 448 attempts / +0.3668 required across
the whole search against 50 attempts / +0.3216 on AUDUSD.r.

What remains true is that the two will always differ, and that the wider one
is the one to quote.

---

## No equities, and therefore no survivorship bias handled

The instruments are FX pairs, metals and oil. There is no equity universe
here, so the engine has never had to deal with delistings, index membership
changes, splits or dividends - and it has **no machinery for any of them**.

This is a limitation of scope, not a strength. If anyone points this engine at
a stock universe, the survivorship-bias question is entirely unhandled and
results will be optimistic in the usual way.

---

## The forward test is short

The live runner has been in dry run on a demo account, on one strategy and one
instrument, for a period measured in days. It is an infrastructure test: it
answers "did the runner see every bar it should have, and decide what the
backtest decided", and it does that well.

It does **not** answer what the strategy costs in reality. In dry run no order
reaches the market, so slippage is not measured at all - the comparison report
says so rather than printing a zero. Nothing here has been validated against
real fills, at any size.

The strategy being forward-tested has no edge and is not expected to. It was
the best cell of a 600-attempt campaign at +0.1990 per trade against a
required +0.4415, which is a fact about the search, not a candidate.

---

## Only technical strategies, and only simple ones

Ten classic rule-based strategies over price and volume. No fundamentals, no
order-flow, no cross-asset signals, no regime models, no machine learning.

The rules are also structurally simple: one position at a time, entry and exit
on bar closes, stops and targets fixed at entry. There is no trailing stop, no
scaling in or out, no portfolio construction across instruments. A conclusion
of "these ten rules do not work on these ten instruments" says nothing about
strategies of a different shape.

---

## Ambiguous bars are an assumption where ticks are missing

When a bar contains both the stop and the target, the engine assumes the stop
and reports the band between that and the optimistic resolution. Where ticks
exist the ambiguity is resolved for real; per the table above, that is a
minority of the sample.

On the baseline strategy the band was worth twelve points of final equity -
the difference between -10% and +1.5%. Where the band straddles zero, **the
result is the assumption**, and no amount of precision elsewhere fixes that.

---

## Campaign reproducibility is bounded by the manifest, not by the engine

Each run pins the instrument spec it used. A *campaign* did not: its cells run
at different moments, and `tick_value` drifts between them - observed at
0.8602, then 0.8605, then 0.8613 on the same instrument. Every money column
scales with it, so re-running a campaign produced different numbers for
reasons that had nothing to do with the code.

A campaign now writes a **manifest** freezing every instrument spec, the
server clock and every setting that changes a number, and
`scripts/verify_campaign.py` re-runs from it and diffs the result cell by
cell. Demonstrated: with `tick_value` moved by 0.13% underneath it, the same
campaign re-run *without* the manifest moved its net PnL (−47.53 → −47.59);
re-run *from* the manifest it matched on every compared field, and the drift
was reported rather than absorbed.

What this does **not** cover:

- **Campaigns run before the manifest existed**, which is all of the ones
  whose numbers are quoted here. Their cells were executed against whatever
  the broker was quoting at the time, and they cannot be reproduced exactly.
- **The bars.** A manifest deliberately does not freeze six years of price
  data. A broker that rewrites a candle changes the result, and that shows up
  as a changed data fingerprint on the run rather than as a matching one.
- **The engine.** Freezing inputs is what makes an engine change *visible*;
  it does not make it absent. A verification across two engine versions is
  expected to differ, and the question is whether the differences are the
  ones the changelog declares.

---

## The synthetic fixture proves nothing about markets

`fixtures/data_cache/` exists so the repository runs without a broker. It is a
random walk with a plausible spread, session week and volume profile. It will
produce a Sharpe ratio, a drawdown and a p-value, and all of them describe a
random number generator.

It is labelled `synthetic_fixture` in the cache provenance, announced by the
API in `/api/health`, and bannered on every page of the dashboard. None of the
results quoted in the README were measured on it.

---

## What the engine is actually good for

Given all of the above: this is an instrument for **failing to find an edge
credibly**. It is built so that a negative result can be trusted - the costs
are not understated, the trials are counted, the assumptions are reported as
bands, and the simulator is the same code as the trader.

A positive result out of it would still need everything above addressed before
anyone risked money on it.
