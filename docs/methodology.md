# Methodology

This document is for someone deciding whether to believe a number this engine
produced. It explains the choices that decide results, and why each one was
made. Where a measurement contradicted the expectation that motivated the
feature, the measurement is here and the expectation is named as wrong.

The organising principle is narrow: **the engine must not be able to flatter a
strategy.** Every choice below is downstream of that, and several of them make
reported results worse than a more permissive engine would report.

---

## 1. The unit of time, and why the server clock is the whole game

MetaTrader 5 hands back timestamps as epoch seconds that actually encode the
*broker server's wall clock*, not a UTC instant. A server showing 13:44 at
UTC+3 returns the epoch of 13:44 UTC. Taken at face value, the entire history
shifts by two or three hours. On an intraday strategy that is not a rounding
error - it moves every bar across session boundaries and invalidates any
result that depends on time of day.

So the timezone is measured, not assumed. `core/data/servertime.py` compares
the newest tick against the local clock and quantises to thirty minutes, which
is enough to absorb network latency and not enough to absorb a wrong guess.
The offset resolves to an **IANA zone**, not a fixed offset, because only a
zone reconstructs past DST transitions correctly - a fixed +3 applied across a
March boundary is wrong for half the year.

The inference refuses rather than guesses. A measurement more than two minutes
from a half-hour multiple is rejected, which is what happens when the market
is closed and the "newest" quote is hours old. That refusal is why eight tests
skip on a weekend instead of failing: they have nothing to stand on, and
rounding a stale quote's age into a plausible-looking offset would be the
failure mode this check exists to prevent.

There is a second trap on the way back in. The `MetaTrader5` package converts
*naive* datetimes using the local machine's timezone, so request bounds arrive
silently shifted on any PC not set to UTC - measured at two hours on a UTC+2
machine. Every request therefore passes a tz-aware datetime carrying the
server's wall clock.

---

## 2. The spread field was not the spread

This is the measurement that changed the most, and it contradicted an
assumption the engine had been built on.

Above M1, MetaTrader's `spread` column on a bar is **the minimum of the
spreads of the M1 bars inside that period** - not the average, not the spread
at any particular moment. Measured against the M1 bars underneath them, this
matched in **100.0% of more than four thousand hourly periods on each of three
instruments**.

Charged as a transaction cost, that field bills the best price of the period.
The size of the error is not marginal:

| instrument | charged from the H1 field | measured from M1 |
|---|---|---|
| WTI H1 | 4 points | 29 points |
| four FX majors, H1 | **0** on 67-76% of bars | non-zero throughout |

Backtests run that way were trading for free on most bars. The correction is
structural rather than a warning:

- The raw field is renamed `min_spread_m1` at every timeframe above M1
  (`core/data/provider.py`). Two different quantities are no longer allowed to
  share one name, because a column called `spread` gets charged as a fill cost
  eventually.
- The cost model **refuses** to charge `min_spread_m1`. A run that asks for a
  per-bar spread above M1 either gets one reconstructed from the M1 sample of
  the same period, or does not start.
- Every run reports what it actually charged, so the question "what spread was
  this?" has an answer on the result rather than in someone's memory.

**The honest limitation:** reconstruction needs M1 bars for the period, and
the broker does not keep M1 forever. Where M1 is unavailable, a fixed spread
is charged, and it is a fixed *recent* spread applied to older bars. See
`limitations.md`.

---

## 3. Ambiguous bars, and the band that matters more than the result

When a bar's range contains both the stop and the target, bar data cannot say
which was touched first. The engine assumes the **stop** - the pessimistic
choice - and flags the trade.

Assuming is not the interesting part. Reporting the size of the assumption is.
Every run carries an **uncertainty band**: the result with every ambiguous bar
resolved as a stop, and with every one resolved as a target. On the baseline
strategy that band was worth twelve points of final equity - the difference
between **-10% and +1.5%**. The strategy was neither profitable nor
unprofitable; the *assumption* was.

A single number in that situation is not a conservative result, it is a
misleading one, and a band is the only honest presentation. Where tick data
exists, `core/validation/tick_resolve.py` settles the ambiguous bars for real
and the band collapses to what was actually measured. Where it does not, the
band stays open and says so.

---

## 4. Counting the attempts, not the successes

Three hundred backtests are three hundred chances to be lucky. At the 5% level
about fifteen come back "significant" with no edge anywhere in the data. An
engine that reports the best cell it found, without saying how many it looked
at, is a machine for manufacturing false discoveries.

So a campaign counts its own attempts, and the count includes **every cell
that was started** - including cells stopped at gate zero before a backtest
ever ran. Stopping early is still a look at the data.

Two thresholds are reported next to the best observed value:

- **Expected maximum under the null**: what the best of N attempts reaches by
  luck alone, given the observed variance across trials.
- **Required value**: what the observed best must clear to be credible at 95%
  confidence after the correction for N attempts.

On the campaign committed in `campaigns/`: 600 cells, of which 70 are refused
at stage zero, leaving **530 attempts**. Best observed **+0.3307** per trade
(ma-crossover / XTIUSD / H1 over 2025, 32 trades, spread measured on 74.9% of
its bars), expected-by-luck **+0.3024**, required **+0.6254**. The best result
is barely above the level a random search of the same size reaches for free,
and well below what it would have to clear. Zero cells survive Bonferroni.

**The maximum has to come from the same search the count does.** A campaign
run in two sittings used to count both halves in N and then take its maximum
over the half it was running, which reported a *smaller* best than the search
had produced and made the winner depend on where the operator stopped for the
night. Carried cells are now candidates as well as observations, and each
carries its own trade count, because the required threshold divides by it.

### Where this went wrong once

The deflated-Sharpe correction was, for one campaign, fed the per-trade Sharpe
of cells with two trades. A two-trade Sharpe reaches ±10 by arithmetic alone,
which inflated the variance across trials and pushed the required threshold to
**+4.77** - a level nothing could clear.

A threshold nothing can clear is not a strict test, it is a broken one, and
it is worth being explicit that a broken test *looks* like rigour. The
variance is now estimated over cells with enough trades to have a meaningful
Sharpe, and the minimum-trade filter is reported with the result.

---

## 5. Gate zero: does the signal beat the spread at all?

Before any strategy runs on an instrument, `core/research/edge.py` asks
whether the raw signal's forward move exceeds the broker's cut, and
`core/research/tradability.py` asks whether the spread is too large a share of
the intended stop for the pair to be tradable at all.

This is a cheap filter that saves expensive campaigns, but the reason it is
first is different: a strategy that looks profitable on an instrument whose
spread eats the stop is not a discovery about the strategy. Finding that out
after the backtest invites explaining it away.

Gate rejections are counted **per gate**, and each reports the share of
signals it rejected. That number was itself a finding: one risk gate was
quietly discarding half of all signals, which is invisible in a result that
only reports the trades that survived.

---

## 6. ATR exits made the dispersion worse

Normalising stop and target distances by ATR was added so that instruments
with different volatilities could be compared on the same scale. Measured, it
**increased** the dispersion of results across instruments rather than
reducing it.

The feature stayed, because per-trade volatility sizing is a legitimate thing
to want to test, and the finding is recorded next to it. It is here because it
is the clearest example of the rule this project runs on: the measurement
outranks the reason the feature was built, and a feature that failed its
stated purpose does not get quietly re-described as having had a different
purpose.

---

## 7. One execution engine, and the test that proves it

The previous version of this project died of a specific bug: the simulator and
the live trader drifted apart, and every reported number described a system
that had never run.

The structural answer is that there is one implementation.
`core/engine/execution.py` holds a single state machine. The backtester drives
it over a DataFrame; the live runner drives it one bar at a time. The rules it
enforces are the same in both:

- signal on the close of bar *t*, execution at the open of *t+1*
- a gap at the open fills at the price that exists, not at the level
- stop before target when both are inside the bar

`tests/test_replay_equivalence.py` replays historical bars through the live
runner and demands trades identical to the backtest - same timestamps, prices,
levels, lots and exit reasons, compared exactly, with no tolerance. Both sides
run the same arithmetic in the same order, so anything other than bit equality
means something moved.

### The false alarm this produced, and what it actually was

A replay diary compared at **+0.0517** against a backtest of the same period,
in a dry run where no order was sent and the difference can only be zero. It
read as an execution divergence - the exact failure the project exists to
prevent - and the plausible explanation was that the diary predated the
unification of execution.

It was not that. Every decision matched to the bit: same entry and exit
timestamps, prices, levels, lots, exit reasons, bars held. Only the money
columns differed, and all of them by **one identical factor,
1.000352809568884**, to within a single ULP across every trade. That factor is
a ratio of `tick_value`: 0.8602076541277064 when the diary was written,
0.8605111436193099 when it was compared. Re-running the current engine with
the older `tick_value` reproduces the old diary to one ULP.

The engine was not the problem. **The comparison was against a different
instrument.** The diary had not pinned the spec it traded, so the comparison
re-read today's spec, and every money column - which is `tick_value / tick_size`
times a price difference - was rescaled on one side of the diff. The residue
landed on the line labelled "slippage", where it is indistinguishable from a
real cost.

The fix is in three places, and the shape of it is the general lesson:

1. The diary now pins the full instrument spec and its cost hash alongside the
   engine version.
2. `compare_live` uses the diary's pinned spec, and reports a spec mismatch or
   a version mismatch as **NOT COMPARABLE** rather than absorbing it into a
   total. It refuses outright to compare across engine versions unless asked
   twice.
3. A test asserts that a dry-run diary from the current engine compares to its
   own backtest at **exactly zero** - `== 0.0`, not `approx`. A tolerance
   would have hidden this.

The same class of error had already been caught once, in the golden reference,
where a regeneration produced 548 differences that were all `tick_value`
drifting from 0.8630 to 0.8612 and none of them anything the engine had done.
It had been fixed there and not generalised. **An input that decides a result
must be pinned by whatever records the result** - a run, a diary, a campaign.
Anything that re-reads it later is comparing two different worlds.

---

## 8. Pinning inputs

`tick_value` moves with an FX rate. Brokers change swap rates without notice -
the same batch once returned XTIUSD at −21.05 and −19.86 under identical
hashes because the rates had changed between the two runs.

So the cost-bearing fields of an instrument spec are part of a run's
**identity**: they are hashed into the `run_id` and stored with the run.
Descriptive fields (digits, currency, trade mode, name) are deliberately left
out, so a broker relabelling something does not invalidate every stored run.

The same applies to the data. A run stores a fingerprint of the actual bar
bytes, not just the date range, because a broker that rewrites a candle
produces a different result over an identical range.

---

## 9. What is deliberately not done

- **No optimiser.** There is no search that returns "the best parameters".
  Parameters are declared in a spec, and where they are canonical (RSI 14 with
  30/70 thresholds is what Wilder published) that is stated in the spec's own
  description. A parameter search is a trial count, and a trial count has to
  be paid for.
- **No implicit look-ahead convenience.** A test recomputes signals over
  truncated history and demands the same values, so no rule can see forward
  through an indicator's warm-up.
- **No result without its observation count.** A metric computed on twenty
  trades reports that it was computed on twenty trades. A test without the
  power to conclude says so rather than returning a number.
- **No cost presented as measured when it was assumed.** Every run and every
  campaign cell reports the share of its bars that had an M1 sample to
  measure the spread on. The campaign's own best cell reads 0.0%.
- **No trading from the research half.** It cannot send an order; there is a
  test asserting the data layer does not even reference the order-sending API.

---

## 10. Reading a result from this engine

In order:

1. **How many attempts produced it?** If it is the best of many, compare it to
   the expected-maximum line, not to zero.
2. **How many trades?** Twenty trades is not evidence of anything.
3. **What spread was charged, and was it measured or assumed?** An edge
   smaller than the difference between those two is not an edge.
4. **How wide is the ambiguity band?** If it straddles zero, the result is the
   assumption.
5. **Is it out-of-sample?** In-sample is not validation.

The engine reports all five without being asked. That is the product.
