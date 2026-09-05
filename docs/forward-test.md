# The forward test

A backtest is a claim about a system that never ran. This is that same system,
running, on bars nobody has seen before, writing down every decision as it
makes it — so that "the simulated system and the live system are the same
system" stops being a property of the test suite and becomes a property of a
record anyone can read.

**The strategy under test has no edge and is not expected to make money.**
`rsi-mean-reversion` on `AUDUSD.r` H4 is the fourth cell of a 530-attempt
search at +0.1990 Sharpe per trade, against the +0.6254 that search's size
demands. It is here because it trades often enough to produce a record —
roughly 172 trades over five and a half years, so a handful a month — and
because a forward test of a strategy chosen for its results would be a test of
nothing. What is under test is the infrastructure.

It runs in **dry run**, on a **demo** account. The broker guard refuses to send
an order on anything but a demo even when sending is explicitly asked for, and
this test does not ask.

---

## Starting it

```powershell
./scripts/forward_test.ps1 -Start
```

That is the whole thing. It launches `scripts.run_live` detached, records the
process id, and puts the diary and the process log side by side in
`logs/live/`:

| file | what it holds |
|---|---|
| `rsi-mean-reversion-AUDUSD.r-H4.jsonl` | the diary: one JSON object per decision |
| `rsi-mean-reversion-AUDUSD.r-H4.out.log` | the process's own account of itself |
| `rsi-mean-reversion-AUDUSD.r-H4.lock` | the PID lock |
| `rsi-mean-reversion-AUDUSD.r-H4.supervisor.pid` | what `-Stop` and `-Status` read |

### Why it is not started with `Start-Process`

Because that does not detach far enough, and the difference was measured
rather than assumed. A caller running inside a Windows **job object** — a CI
agent, a task runner, an automation harness — has its entire process tree
killed when the job closes, and `Start-Process` puts the child in that same
tree. Started that way from inside such a harness, the runner was gone the
moment the shell that launched it was reaped, while `-Status` had reported it
running seconds earlier.

The process is therefore created through WMI (`Win32_Process.Create`), which
has the WMI provider host spawn it, so it inherits none of the caller's job.
It also writes its own log via `--log-file` rather than having a shell
redirect for it: a `cmd /c ... > log` wrapper would be a second process that
also has to survive for weeks, and the PID recorded would be the wrapper's
rather than the runner's.

If WMI refuses, the script says so loudly and falls back to `Start-Process`,
which works fine from an ordinary interactive shell and is only unsafe under
a harness.

Running `-Start` twice is safe: the second one sees the recorded process and
does nothing, and even if it did not, the runner takes a PID lock on the diary
and a second instance refuses to start rather than double the position.

Other instrument, other strategy:

```powershell
./scripts/forward_test.ps1 -Start -Strategy strategies/macd-signal.json -Symbol XAUUSD.r -Timeframe H1
```

**After a reboot it has to be started again.** Nothing here installs a service;
that is a deliberate limit, not an oversight — a trading process that starts
itself on boot is a thing to decide on purpose. To have Windows do it, the
one-liner is:

```powershell
schtasks /create /tn "backtest-engine forward test" /sc onlogon /rl highest ^
  /tr "powershell -NoProfile -ExecutionPolicy Bypass -File \"<repo>\scripts\forward_test.ps1\" -Start"
```

The runner needs the MetaTrader 5 terminal running and logged in; it is the
only external dependency.

## Checking on it

```powershell
./scripts/forward_test.ps1 -Status
```

```
forward test: running (pid 1876, up 14h 22m)
  diary : 318 events, 96.4 KB, last written 09/05/2026 13:00:11
```

`last written` is the number to read. On H4 a bar closes every four hours
during the week, so a diary that has not been written to in more than four
hours *while the market is open* means the runner is up and not receiving —
check the terminal.

## Reading the result

```powershell
./scripts/forward_test.ps1 -Report
```

or, with more control:

```
python -m scripts.compare_live logs/live/rsi-mean-reversion-AUDUSD.r-H4.jsonl \
    --json .temp/forward-report.json
```

It works out which period the diary covers, fetches exactly those bars, runs a
backtest of the same spec with the same costs over exactly that window, and
diffs the two records trade by trade. It can be run at any moment, including
while the runner is going: it opens the diary read-only and touches nothing.

```
Forward test - rsi-mean-reversion on AUDUSD.r H4
  mode              : DRY RUN
  engine at start   : 4.0.0
  diary covers      : 2026-09-07 01:00:00+00:00 -> 2026-10-02 17:00:00+00:00  (25.7 days)
  bars in the diary : 154
  bars replayed     : 154 (from the MT5 terminal)
  process restarts  : 2
  spread charged    : fixed 3.0

Expected vs realized - AUDUSD.r H4
  period            : ...
  trades            : 4 expected, 4 realized, 4 matched
  PnL               : -2.31 expected, -2.31 realized (+0.00)
    of which slippage : +0.00
    unmatched trades  : +0.00
    unexplained       : +0.00
```

### What each line means

**`bars in the diary` against `bars replayed`.** The first is how many closed
bars the runner actually saw; the second is how many exist in that window. They
must be equal. A gap means the runner was down while the market was open, and
every trade it missed will show up below as "only expected" — not as a strategy
result.

**`process restarts`.** How many times a new process took over. More than one
is normal over weeks; each restart re-reads the open position from the broker
by magic number and the closed trades from the diary, so the account continues
rather than resetting to the initial equity.

**`trades: N expected, M realized, K matched`.** Two records hold the same trade
when they entered on the same bar. `K < N` is the interesting case, and every
unmatched trade is printed with the reason the diary gives: an order refused, a
gate that fired live and not in simulation, or a bar the runner never saw.

**The PnL decomposition.** Not one number, three:

- *slippage* — trades both records hold, whose PnL differs. Live, this is the
  price the book gave against the price the engine expected.
- *unmatched trades* — the PnL of trades only one side took.
- *unexplained* — the rest. **This is the line to read first.** It should be
  zero. Anything else is a defect nobody has found yet.

**In dry run, "slippage" should also be zero.** No order was sent, so both
sides are the same engine's arithmetic over the same bars, and a non-zero value
there is not a cost — it is the runner and the backtester disagreeing. The
report says so explicitly when it happens, and names the three inputs that can
cause it: the spread charged, bars the broker has rewritten since, and the
engine version between the diary and the backtest.

### What it cannot tell you

- **Whether the strategy works.** It does not, and four trades a month will not
  settle that either way. Ten trades is not a sample; the campaign's own
  threshold is thirty, and its verdict on this cell was already negative.
- **Real slippage, real rejections, real partial fills.** Dry run reaches no
  order book. Turning that on is `python -m scripts.run_live ... --send`, typed
  deliberately, and the broker still refuses anything but a demo account.

## Stopping it

```powershell
./scripts/forward_test.ps1 -Stop
```

The process is stopped outright. It holds no unflushed state: the diary is
fsynced line by line, and an open position belongs to the broker rather than to
the process — the next start reads both back. **Stopping does not close an open
position**, on purpose: a process that liquidated on exit would make every
restart cost money.

## What it survives, and how

| what happens | what the runner does |
|---|---|
| the terminal disconnects | the fetch fails, goes in the diary, and the wait doubles from 20s to a 5-minute ceiling until it comes back. Nothing is decided while disconnected, because no bar arrives to decide on |
| the process dies | the lock is released; the next start reconciles the position from the broker by magic number and the equity from the diary |
| the shell that started it is killed | nothing: the process was spawned by the WMI service and belongs to no job object of that shell's. Verified by launching it from a harness task and checking it was still running after the task was reaped |
| a foreign position appears on the symbol | the runner **refuses to start**. Sizing and risk here assume this engine owns the exposure, and adopting a trade whose stop it never chose would be worse than stopping |
| the weekend | no closed bar arrives, so nothing happens. The session calendar counts the gap rather than the rows, so Monday's bar is one session bar after Friday's |
| a DST change | the session calendar is pinned once and carries the server timezone, mapping each UTC instant to its weekly slot **on the server clock**. Neither the session nor the bar a time stop fires on moves. This is the phase-7 A2 fix, and this is the case it was written for |
| the account is not a demo | the broker guard refuses to send, whatever the flags say |

## Where the numbers come from

- **The spread** is the instrument's median measured on **M1**, where the
  field is a spread. It is charged as a constant rather than read off the bar,
  because above M1 the broker's field is the minimum of the M1 spreads inside
  the bar — `min_spread_m1` — and charging that means charging the best price
  of the period. On AUDUSD.r that is 3 points, against a column whose median is
  0 on 77% of daily bars.
- **The session calendar** is inferred once from the warm-up history on the
  server clock and pinned, so the time stop counts the same bars a backtest of
  the same period counts.
- **The equity** starts at 100 and is carried across restarts.
