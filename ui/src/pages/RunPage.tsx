import { DataUpdate } from "../components/DataUpdate";
import { NumberField } from "../components/NumberField";
import { Progress } from "../components/Progress";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type BacktestResponse,
  type BreakevenPrior,
  type Coverage,
  type EdgeReport,
  type RunConfigIn,
  type RunSummary,
  type Strategy,
  type SymbolList,
} from "../api/client";
import {
  Badge,
  Empty,
  ErrorNotice,
  Field,
  Loading,
  Notice,
  Panel,
  Signed,
  StatusBadge,
} from "../components/ui";
import { int, isoDateInput, num, pct, signedMoney, utcDate, utcDateTime } from "../format";
import { navigate } from "../router";

const TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"] as const;
const SPREAD_MODES = [
  { value: "per_bar", label: "per bar (from the feed)" },
  { value: "fixed", label: "fixed (points)" },
  { value: "quantile", label: "quantile (0–1)" },
] as const;

function toIso(day: string): string | null {
  return day ? `${day}T00:00:00Z` : null;
}

export function RunPage() {
  const [symbols, setSymbols] = useState<SymbolList | null>(null);
  const [strategies, setStrategies] = useState<Strategy[] | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);

  const [strategyId, setStrategyId] = useState("");
  const [symbol, setSymbol] = useState("");
  const [timeframe, setTimeframe] = useState<string>("M1");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [equity, setEquity] = useState(100);
  const [spreadMode, setSpreadMode] = useState<RunConfigIn["spread_mode"]>("per_bar");
  const [spreadValue, setSpreadValue] = useState<number | "">("");
  const [commission, setCommission] = useState(0);

  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [coverageBusy, setCoverageBusy] = useState(false);
  // bumped when a download finishes, so the coverage is re-read without
  // duplicating the effect that reads it
  const [coverageNonce, setCoverageNonce] = useState(0);
  const reloadCoverage = () => setCoverageNonce((n) => n + 1);

  const [edge, setEdge] = useState<EdgeReport | null>(null);
  const [edgeBusy, setEdgeBusy] = useState(false);
  const [edgeError, setEdgeError] = useState<unknown>(null);

  const [launch, setLaunch] = useState<BacktestResponse | null>(null);
  const [runBusy, setRunBusy] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);
  const [recent, setRecent] = useState<RunSummary[]>([]);
  const pollTimer = useRef<number | null>(null);

  useEffect(() => {
    Promise.all([api.symbols(), api.strategies()])
      .then(([symbolList, strategyList]) => {
        setSymbols(symbolList);
        setStrategies(strategyList);
        const first = strategyList[0];
        if (first) {
          setStrategyId(first.id);
          setTimeframe(first.timeframe);
          // A strategy names the instrument it was written for, and that
          // instrument need not be in this cache - on a fresh clone serving
          // the synthetic fixture it never is. Selecting it anyway leaves
          // the <select> showing its first option while the state holds
          // something else, so the page reports "no data" for a symbol the
          // user can see is selected. Fall back to what is actually there.
          const available = symbolList.symbols.map((entry) => entry.name);
          const cached = symbolList.cached_symbols ?? [];
          setSymbol(
            available.includes(first.symbol)
              ? first.symbol
              : (cached[0] ?? available[0] ?? first.symbol),
          );
        }
      })
      .catch(setLoadError);
    void refreshRecent();
    return () => {
      if (pollTimer.current) window.clearInterval(pollTimer.current);
    };
  }, []);

  // Whether the M1 sample spans the bars being run. Dates rather than counts:
  // M1 is a different number of bars by construction, and what matters is
  // whether it reaches both ends of the period.
  function coversTheSamePeriod(bars: Coverage, minutes: Coverage | null): boolean {
    if (!minutes || !minutes.start || !minutes.end || !bars.start || !bars.end) {
      return false;
    }
    return (
      Date.parse(minutes.start) <= Date.parse(bars.start) &&
      Date.parse(minutes.end) >= Date.parse(bars.end)
    );
  }

  async function refreshRecent() {
    try {
      setRecent(await api.runs({ limit: 8 }));
    } catch {
      /* the recent list is a bonus: without it the page stays usable */
    }
  }

  // Cached coverage decides the bounds of the period: asking for dates that
  // do not exist is the most common way to get "no data".
  //
  // It also decides the spread policy. Per-bar above M1 is rebuilt from the
  // M1 bars of the same period, and the engine refuses the run outright when
  // they are not there - correctly, but as the *default* on a cache holding
  // only hourly bars it means the first backtest anyone tries fails. So the
  // default follows the data: per bar where M1 covers the period, a measured
  // constant where it does not. Either can still be chosen by hand.
  useEffect(() => {
    if (!symbol) return;
    let cancelled = false;
    setCoverageBusy(true);
    Promise.all([
      api.coverage(symbol, timeframe),
      timeframe === "M1"
        ? Promise.resolve(null)
        : api.coverage(symbol, "M1").catch(() => null),
    ])
      .then(([data, minutes]) => {
        if (cancelled) return;
        setCoverage(data);
        setStart(isoDateInput(data.start));
        setEnd(isoDateInput(data.end));
        if (coversTheSamePeriod(data, minutes)) {
          setSpreadMode("per_bar");
        } else {
          // the measured median, not a guess: it is what the instrument's
          // own M1 sample says, and the panel states the period it came from
          setSpreadMode("fixed");
          setSpreadValue(minutes?.spread_median_points ?? data.spread_median_points ?? "");
        }
      })
      .catch(() => !cancelled && setCoverage(null))
      .finally(() => !cancelled && setCoverageBusy(false));
    return () => {
      cancelled = true;
    };
  }, [symbol, timeframe, coverageNonce]);

  const strategy = useMemo(
    () => strategies?.find((item) => item.id === strategyId) ?? null,
    [strategies, strategyId],
  );

  const config: RunConfigIn = {
    symbol,
    timeframe,
    start: toIso(start),
    end: toIso(end),
    initial_equity: equity,
    spread_mode: spreadMode,
    spread_value: spreadValue === "" ? null : Number(spreadValue),
    commission_per_lot_per_side: commission,
    swap_mode: "points",
    session_threshold: 0.5,
    // above M1 a per-bar spread is rebuilt from the M1 sample; the
    // median is the point of that distribution a fill is charged at
    per_bar_spread_quantile: 0.5,
  };

  const hasData = (coverage?.bars ?? 0) > 0;
  const disabled = !strategyId || !symbol || !hasData || edgeBusy || runBusy;

  async function onCheckEdge() {
    setEdgeBusy(true);
    setEdgeError(null);
    setEdge(null);
    try {
      setEdge(await api.edge({ strategy_id: strategyId, config }));
    } catch (error) {
      setEdgeError(error);
    } finally {
      setEdgeBusy(false);
    }
  }

  function startPolling(runId: string) {
    if (pollTimer.current) window.clearInterval(pollTimer.current);
    pollTimer.current = window.setInterval(async () => {
      try {
        const detail = await api.run(runId);
        if (detail.status === "running") return;
        window.clearInterval(pollTimer.current!);
        pollTimer.current = null;
        setRunBusy(false);
        void refreshRecent();
        if (detail.status === "done") navigate(`/result/${runId}`);
        else setRunError(new Error(detail.error ?? "the run failed"));
      } catch (error) {
        window.clearInterval(pollTimer.current!);
        pollTimer.current = null;
        setRunBusy(false);
        setRunError(error);
      }
    }, 800);
  }

  async function onRunBacktest() {
    setRunBusy(true);
    setRunError(null);
    setLaunch(null);
    try {
      const response = await api.backtest({ strategy_id: strategyId, config });
      setLaunch(response);
      if (response.status === "running") {
        startPolling(response.run_id);
      } else {
        setRunBusy(false);
        void refreshRecent();
        if (response.status === "done") navigate(`/result/${response.run_id}`);
      }
    } catch (error) {
      setRunError(error);
      setRunBusy(false);
    }
  }

  if (loadError !== null) return <ErrorNotice error={loadError} />;
  if (!symbols || !strategies) return <Loading label="Loading symbols and strategies…" />;
  if (strategies.length === 0) {
    return (
      <Panel title="No strategy">
        <Empty title="The strategies/ folder is empty">
          <p>
            Add a strategy JSON file to <code>strategies/</code> and reload the page.
          </p>
        </Empty>
      </Panel>
    );
  }

  const cached = new Set(symbols.cached_symbols);

  return (
    <div className="stack">
      <Panel
        title="Run configuration"
        aside={
          <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
            strategy parameters live in the JSON, this is only execution
          </span>
        }
      >
        <div className="stack">
          <div className="form-grid">
            <Field
              label="Strategy"
              htmlFor="f-strategy"
              hint={strategy ? `${strategy.file}` : undefined}
            >
              <select
                id="f-strategy"
                value={strategyId}
                onChange={(event) => {
                  const next = strategies.find((item) => item.id === event.target.value);
                  setStrategyId(event.target.value);
                  if (next) {
                    setSymbol(next.symbol);
                    setTimeframe(next.timeframe);
                  }
                }}
              >
                {strategies.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} ({item.id})
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="Symbol"
              htmlFor="f-symbol"
              hint={cached.has(symbol) ? "data in cache" : "no data downloaded for this symbol"}
            >
              <select
                id="f-symbol"
                value={symbol}
                onChange={(event) => setSymbol(event.target.value)}
              >
                {symbols.symbols.map((item) => (
                  <option key={item.name} value={item.name}>
                    {cached.has(item.name) ? "● " : "○ "}
                    {item.name}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Timeframe" htmlFor="f-tf">
              <select
                id="f-tf"
                value={timeframe}
                onChange={(event) => setTimeframe(event.target.value)}
              >
                {TIMEFRAMES.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label="From (UTC)"
              htmlFor="f-start"
              hint={coverage?.start ? `cached from ${utcDate(coverage.start)}` : "—"}
            >
              <input
                id="f-start"
                type="date"
                value={start}
                min={isoDateInput(coverage?.start)}
                max={isoDateInput(coverage?.end)}
                onChange={(event) => setStart(event.target.value)}
              />
            </Field>

            <Field
              label="To (UTC)"
              htmlFor="f-end"
              hint={coverage?.end ? `cached until ${utcDate(coverage.end)}` : "—"}
            >
              <input
                id="f-end"
                type="date"
                value={end}
                min={isoDateInput(coverage?.start)}
                max={isoDateInput(coverage?.end)}
                onChange={(event) => setEnd(event.target.value)}
              />
            </Field>

            <Field label="Initial equity" htmlFor="f-equity" hint="account currency">
              <NumberField
                id="f-equity"
                min={1}
                step={1}
                value={equity}
                onChange={(value) => setEquity(value ?? 0)}
              />
            </Field>

            <Field label="Spread policy" htmlFor="f-spread">
              <select
                id="f-spread"
                value={spreadMode}
                onChange={(event) => {
                  setSpreadMode(event.target.value as RunConfigIn["spread_mode"]);
                  setSpreadValue("");
                }}
              >
                {SPREAD_MODES.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field
              label={spreadMode === "quantile" ? "Quantile" : "Spread (points)"}
              htmlFor="f-spread-value"
              hint={
                spreadMode === "per_bar"
                  ? "uses the spread column of the feed"
                  : spreadMode === "quantile"
                    ? "e.g. 0.9 = ninetieth percentile"
                    : "fixed cost on every fill"
              }
            >
              <NumberField
                id="f-spread-value"
                step={spreadMode === "quantile" ? 0.05 : 1}
                min={0}
                max={spreadMode === "quantile" ? 1 : undefined}
                disabled={spreadMode === "per_bar"}
                value={spreadValue === "" ? null : spreadValue}
                onChange={(value) => setSpreadValue(value ?? "")}
              />
            </Field>

            <Field label="Commission" htmlFor="f-commission" hint="per lot, per side">
              <NumberField
                id="f-commission"
                min={0}
                step={0.5}
                value={commission}
                onChange={(value) => setCommission(value ?? 0)}
              />
            </Field>
          </div>

          {coverageBusy && <Loading label="Reading the cached coverage…" height={24} />}
          {!coverageBusy && coverage && !hasData && (
            <Notice kind="warn" title="No cached data for this combination">
              Nothing to backtest on yet. Download it from the broker in the
              panel below, or from a terminal with{" "}
              <code>
                python -m examples.download_year {symbol} --timeframe {timeframe}
              </code>
              .
            </Notice>
          )}
          {!coverageBusy && coverage && hasData && (
            <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>
              Cache: <strong>{int(coverage.bars)}</strong> {coverage.timeframe} bars from{" "}
              <strong>{utcDate(coverage.start)}</strong> to{" "}
              <strong>{utcDate(coverage.end)}</strong>
              {coverage.quality && (
                <>
                  {" · completeness "}
                  <strong>{pct(coverage.quality.completeness, 2)}</strong>
                  {" · "}
                  {int(coverage.quality.gaps)} session gaps
                </>
              )}
            </div>
          )}

          <div className="actions">
            <button onClick={onCheckEdge} disabled={disabled}>
              {edgeBusy ? "Measuring the edge…" : "Check edge"}
            </button>
            <button className="primary" onClick={onRunBacktest} disabled={disabled}>
              {runBusy ? "Backtest running…" : "Run backtest"}
            </button>
            <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
              Gate zero is fast and says whether the backtest is worth running.
            </span>
          </div>
        </div>
      </Panel>

      <Panel
        title="Broker data"
        aside={
          <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
            MetaTrader 5 has to be running and logged in
          </span>
        }
      >
        <DataUpdate
          symbol={symbol}
          timeframe={timeframe}
          coverage={coverage}
          onFinished={reloadCoverage}
        />
      </Panel>

      {edgeError !== null && <ErrorNotice error={edgeError} />}
      {edgeBusy && (
        <Panel title="Gate zero">
          <Loading label="Computing the forward returns…" height={120} />
        </Panel>
      )}
      {edge && !edgeBusy && <EdgePanel report={edge} />}

      {runError !== null && <ErrorNotice error={runError} />}
      {launch && runBusy && (
        <Panel title="Backtest running">
          <Progress
            label={`${strategyId} on ${symbol} ${timeframe}`}
            detail={
              <>
                {launch.message} — run <span className="mono">{launch.run_id}</span>. The
                result page opens by itself when it finishes.
              </>
            }
          />
        </Panel>
      )}

      <RecentRuns runs={recent} onDeleted={refreshRecent} />
    </div>
  );
}

function EdgePanel({ report }: { report: EdgeReport }) {
  return (
    <Panel
      className="reveal"
      title="Gate zero — does the signal beat the spread?"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          {int(report.signals_long)} long signals · {int(report.signals_short)} short ·{" "}
          {int(report.bars)} bars
        </span>
      }
      tight
    >
      <div style={{ padding: "16px 16px 0" }}>
        <Notice kind={report.passed ? "ok" : "warn"} title={report.passed ? "PASS" : "NO PASS"}>
          {report.verdict}
        </Notice>
      </div>

      {report.breakeven_prior && <BreakevenPriorBlock prior={report.breakeven_prior} />}

      <div className="table-scroll" style={{ marginTop: 16 }}>
        <table>
          <caption>
            Forward return from the signal: entry at the open of the next bar, exit at the
            close after N bars. Values in instrument points.
          </caption>
          <thead>
            <tr>
              <th scope="col" className="num">
                Horizon
              </th>
              <th scope="col">Direction</th>
              <th scope="col" className="num">
                Obs.
              </th>
              <th scope="col" className="num">
                Mean
              </th>
              <th scope="col" className="num">
                Std err
              </th>
              <th scope="col" className="num">
                t
              </th>
              <th scope="col" className="num">
                p
              </th>
              <th scope="col" className="num">
                Drift
              </th>
              <th scope="col" className="num">
                Spread
              </th>
              <th scope="col" className="num">
                Net
              </th>
              <th scope="col">Verdict</th>
            </tr>
          </thead>
          <tbody>
            {report.stats.map((stat) => (
              <tr
                key={`${stat.horizon}-${stat.direction}`}
                className={stat.beats_cost ? "flagged" : undefined}
              >
                <td className="num">{stat.horizon}</td>
                <td>{stat.direction}</td>
                <td className="num">{int(stat.observations)}</td>
                <td className="num">
                  <Signed value={stat.mean_points} text={num(stat.mean_points, 2)} />
                </td>
                <td className="num">{num(stat.stderr_points, 2)}</td>
                <td className="num">{num(stat.t_stat, 2)}</td>
                <td className="num">{num(stat.p_value, 4)}</td>
                <td className="num">{num(stat.drift_baseline_points, 2)}</td>
                <td className="num">{num(stat.spread_cost_points, 1)}</td>
                <td className="num">
                  <strong>
                    <Signed value={stat.net_points} text={signedMoney(stat.net_points, 2)} />
                  </strong>
                </td>
                <td>
                  {stat.beats_cost ? (
                    <Badge kind="ok">beats the costs</Badge>
                  ) : (
                    <span style={{ color: "var(--ink-faint)", whiteSpace: "normal" }}>
                      {stat.verdict}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/**
 * A-priori break-even win rate, from the spec alone: the share of trades that
 * has to win before the strategy stops losing money. Known before any
 * backtest runs, it is the target the realized win rate has to clear.
 */
function BreakevenPriorBlock({ prior }: { prior: BreakevenPrior }) {
  return (
    <div style={{ padding: "16px 16px 0" }}>
      <h3 style={{ marginBottom: 8 }}>Break-even win rate (a priori)</h3>
      {!prior.valid ? (
        <Notice kind="info" title="Not computable for this spec">
          {prior.reason ?? "the exits are not a fixed stop and target in points"}
        </Notice>
      ) : (
        <>
          <dl className="facts">
            <dt>Required win rate</dt>
            <dd>
              <strong>{pct(prior.breakeven_win_rate)}</strong>
            </dd>
            <dt>Loss leg / win leg</dt>
            <dd>
              {num(prior.loss_points, 1)} / {num(prior.win_points, 1)} points
            </dd>
            <dt>Commission (round turn)</dt>
            <dd>{num(prior.commission_points, 2)} points</dd>
            <dt>Average spread</dt>
            <dd>
              {prior.avg_spread_points === null || prior.avg_spread_points === undefined
                ? "—"
                : `${num(prior.avg_spread_points, 1)} points`}
            </dd>
          </dl>
          {(prior.caveats ?? []).length > 0 && (
            <ul className="caveats">
              {(prior.caveats ?? []).map((caveat) => (
                <li key={caveat}>{caveat}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

function RecentRuns({ runs, onDeleted }: { runs: RunSummary[]; onDeleted: () => void }) {
  const [pending, setPending] = useState<string | null>(null);

  async function remove(runId: string) {
    setPending(runId);
    try {
      await api.deleteRun(runId);
      onDeleted();
    } finally {
      setPending(null);
    }
  }

  return (
    <Panel title="Recent runs" aside={<a href="#/compare">compare →</a>} tight>
      {runs.length === 0 ? (
        <Empty title="No saved run">
          <p>Launch a backtest: the result stays on disk under runs/.</p>
        </Empty>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Strategy</th>
                <th scope="col">Instrument</th>
                <th scope="col">Period (UTC)</th>
                <th scope="col" className="num">
                  Trades
                </th>
                <th scope="col" className="num">
                  Return
                </th>
                <th scope="col" className="num">
                  Sharpe
                </th>
                <th scope="col">Status</th>
                <th scope="col" />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id}>
                  <td className="mono">
                    <a href={`#/result/${run.run_id}`}>{run.run_id.slice(0, 8)}</a>
                  </td>
                  <td>{run.strategy_id}</td>
                  <td>
                    {run.symbol} {run.timeframe}
                  </td>
                  <td>
                    {utcDate(run.data_start)} → {utcDate(run.data_end)}
                  </td>
                  <td className="num">{int(run.trades)}</td>
                  <td className="num">
                    <Signed
                      value={run.total_return}
                      text={run.total_return === null ? "—" : pct(run.total_return)}
                    />
                  </td>
                  <td className="num">{num(run.sharpe, 2)}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td>
                    <button
                      onClick={() => remove(run.run_id)}
                      disabled={pending === run.run_id}
                      aria-label={`Delete run ${run.run_id}`}
                    >
                      delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <div className="pager">
        <span>
          Runs are folders under <code>runs/</code>: spec, trades, equity and metrics on disk.
        </span>
        <span className="spread">
          last refresh {utcDateTime(new Date().toISOString(), true)} UTC
        </span>
      </div>
    </Panel>
  );
}
