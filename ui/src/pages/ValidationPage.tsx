import { NumberField } from "../components/NumberField";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  type MultipleTestingResponse,
  type PermutationResponse,
  type PermutationTest,
  type RunSummary,
  type TickResolveResponse,
  type WalkForwardResponse,
} from "../api/client";
import { Badge, Empty, ErrorNotice, Field, Loading, Notice, Panel, Signed } from "../components/ui";
import { int, money, num, pct, signedMoney, utcDate, utcDateTime } from "../format";

const OOS_COLOR = "#1b4f9c";
const NULL_COLOR = "#9aa4ae";
const OBSERVED_COLOR = "#a4232a";

/**
 * Everything on this page exists to make a backtest harder to believe, not
 * easier. The four blocks attack the same number from four sides: decay out
 * of sample, distance from a null model, the discount owed for the number of
 * attempts, and the assumption made on ambiguous bars.
 */
export function ValidationPage({ runId }: { runId: string }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selected, setSelected] = useState(runId);

  useEffect(() => {
    api
      .runs({ limit: 100 })
      .then((all) => {
        const done = all.filter((run) => run.status === "done");
        setRuns(done);
        if (!selected && done[0]) setSelected(done[0].run_id);
      })
      .catch(() => setRuns([]));
    // the run id in the URL wins over whatever was picked before
    if (runId) setSelected(runId);
  }, [runId]);

  if (!runs) return <Loading label="Loading the runs…" height={220} />;
  if (runs.length === 0) {
    return (
      <Panel title="Validation">
        <Empty title="No completed run to validate">
          <p>
            <a href="#/">Launch a backtest</a> first: validation reads a saved run, it does
            not produce one.
          </p>
        </Empty>
      </Panel>
    );
  }

  return (
    <div className="stack">
      <Panel
        title="Validation"
        aside={
          <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            every block below is a way of trying to disprove the run
          </span>
        }
      >
        <Field label="Run" htmlFor="v-run" hint="only completed runs can be validated">
          <select
            id="v-run"
            value={selected}
            onChange={(event) => {
              setSelected(event.target.value);
              window.history.replaceState(null, "", `#/validation/${event.target.value}`);
            }}
          >
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id.slice(0, 8)} · {run.strategy_id} · {run.symbol} {run.timeframe} ·{" "}
                {utcDate(run.data_start)} → {utcDate(run.data_end)}
              </option>
            ))}
          </select>
        </Field>
      </Panel>

      {selected && (
        <>
          <WalkForwardSection runId={selected} />
          <PermutationSection runId={selected} />
          <MultipleTestingSection runId={selected} />
          <TickResolveSection runId={selected} />
        </>
      )}
    </div>
  );
}

// -- walk-forward ---------------------------------------------------------

function WalkForwardSection({ runId }: { runId: string }) {
  const [mode, setMode] = useState<"rolling" | "anchored">("rolling");
  const [trainDays, setTrainDays] = useState(90);
  const [testDays, setTestDays] = useState(30);
  const [minTrades, setMinTrades] = useState(30);
  const [gridText, setGridText] = useState("");
  const [report, setReport] = useState<WalkForwardResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      let grid: Record<string, unknown[]> | null = null;
      if (gridText.trim()) {
        grid = JSON.parse(gridText) as Record<string, unknown[]>;
      }
      setReport(
        await api.walkForward({
          run_id: runId,
          mode,
          train_days: trainDays,
          test_days: testDays,
          min_train_trades: minTrades,
          grid: grid as never,
        }),
      );
    } catch (problem) {
      setError(
        problem instanceof SyntaxError
          ? new Error(`the grid is not valid JSON: ${problem.message}`)
          : problem,
      );
    } finally {
      setBusy(false);
    }
  }

  const curve = useMemo(
    () =>
      (report?.oos_curve ?? []).map((point) => ({
        ts: new Date(point.t).getTime(),
        equity: point.equity,
      })),
    [report],
  );

  return (
    <Panel
      title="Walk-forward"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          only the concatenated out-of-sample curve may be read as performance
        </span>
      }
    >
      <div className="stack">
        <div className="form-grid">
          <Field label="Window mode" htmlFor="wf-mode">
            <select
              id="wf-mode"
              value={mode}
              onChange={(event) => setMode(event.target.value as "rolling" | "anchored")}
            >
              <option value="rolling">rolling (fixed-length train)</option>
              <option value="anchored">anchored (train grows from the start)</option>
            </select>
          </Field>
          <Field label="Train days" htmlFor="wf-train" hint="in-sample leg">
            <NumberField
              id="wf-train"
              min={1}
              value={trainDays}
              onChange={(value) => setTrainDays(value ?? 1)}
            />
          </Field>
          <Field label="Test days" htmlFor="wf-test" hint="out-of-sample leg, and the step">
            <NumberField
              id="wf-test"
              min={1}
              value={testDays}
              onChange={(value) => setTestDays(value ?? 1)}
            />
          </Field>
          <Field
            label="Min train trades"
            htmlFor="wf-min"
            hint="below this the window is discarded"
          >
            <NumberField
              id="wf-min"
              min={0}
              value={minTrades}
              onChange={(value) => setMinTrades(value ?? 0)}
            />
          </Field>
          <Field
            label="Parameter grid (JSON)"
            htmlFor="wf-grid"
            hint="empty = no optimization"
          >
            <input
              id="wf-grid"
              type="text"
              placeholder={'{"exit.stop_loss.value": [100, 150, 200]}'}
              value={gridText}
              onChange={(event) => setGridText(event.target.value)}
            />
          </Field>
        </div>

        <div className="actions">
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Running the windows…" : "Run walk-forward"}
          </button>
          <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
            Without a grid nothing is selected, so nothing here speaks to overfitting.
          </span>
        </div>

        {error !== null && <ErrorNotice error={error} />}
        {busy && <Loading label="Optimizing each in-sample leg…" height={200} />}

        {report && (
          <>
            {(report.warnings ?? []).map((warning) => (
              <Notice key={warning} kind="warn" title="Read this before the numbers">
                {warning}
              </Notice>
            ))}

            <Notice
              kind={
                (report.oos_metrics?.net_pnl ?? 0) > 0 && report.windows_evaluated > 0
                  ? "ok"
                  : "warn"
              }
              title="Walk-forward verdict"
            >
              {report.verdict}
            </Notice>

            <dl className="facts">
              <dt>Windows</dt>
              <dd>
                {int(report.windows_evaluated)} usable, {int(report.windows_skipped)} discarded
              </dd>
              <dt>Embargo between legs</dt>
              <dd>
                {num(report.embargo_minutes, 0)} minutes (longest holding the spec can
                produce)
              </dd>
              <dt>Grid</dt>
              <dd>
                {report.grid_size} combination{report.grid_size === 1 ? "" : "s"} per window
              </dd>
            </dl>

            {curve.length > 0 && (
              <div>
                <h3>Concatenated out-of-sample equity</h3>
                <div className="legend" style={{ padding: "4px 0" }}>
                  <span>
                    <span className="swatch" style={{ background: OOS_COLOR }} />
                    Equity carried across windows (account currency) — Y axis does not start
                    at zero
                  </span>
                </div>
                <div className="chart-frame" style={{ height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={curve} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
                      <CartesianGrid stroke="#eceff1" />
                      <XAxis
                        dataKey="ts"
                        type="number"
                        scale="time"
                        domain={["dataMin", "dataMax"]}
                        tickFormatter={(value: number) =>
                          utcDate(new Date(value).toISOString())
                        }
                        tick={{ fontSize: 11 }}
                        minTickGap={48}
                      />
                      <YAxis
                        tick={{ fontSize: 11 }}
                        width={64}
                        domain={["auto", "auto"]}
                        tickFormatter={(value: number) => money(value, 0)}
                      />
                      <ReferenceLine
                        y={report.oos_metrics?.initial_equity ?? 100}
                        stroke="#5a646e"
                        strokeDasharray="4 3"
                      />
                      <Tooltip
                        isAnimationActive={false}
                        content={({ active, payload, label }) => {
                          if (!active || !payload?.length) return null;
                          return (
                            <div className="tooltip">
                              <div className="t">
                                {utcDateTime(
                                  new Date((label as number) ?? 0).toISOString(),
                                  true,
                                )}{" "}
                                UTC
                              </div>
                              <dl>
                                <dt>equity</dt>
                                <dd>{money(payload[0]?.value as number, 4)}</dd>
                              </dl>
                            </div>
                          );
                        }}
                      />
                      <Area
                        type="monotone"
                        dataKey="equity"
                        stroke={OOS_COLOR}
                        fill={OOS_COLOR}
                        fillOpacity={0.1}
                        strokeWidth={1.6}
                        isAnimationActive={false}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            <WindowTable report={report} />
            <DegradationTable report={report} />
            <StabilityTable report={report} />
          </>
        )}
      </div>
    </Panel>
  );
}

/**
 * Short label for a dotted spec path. The last segment is usually the
 * generic "value", which would render two different parameters identically:
 * drop it, and drop the leading block, leaving the part that names the thing.
 */
function shortParam(key: string): string {
  const parts = key.split(".").filter((part) => part !== "value");
  return parts.length > 1 ? parts.slice(1).join(".") : parts.join(".") || key;
}

function WindowTable({ report }: { report: WalkForwardResponse }) {
  return (
    <div className="table-scroll">
      <table>
        <caption>
          One row per window. In-sample figures are the best of {report.grid_size} candidate
          {report.grid_size === 1 ? "" : "s"} on that data, so they are an upper bound by
          construction; only the out-of-sample column is a measurement.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="num">
              #
            </th>
            <th scope="col">Train (UTC)</th>
            <th scope="col">Test (UTC)</th>
            <th scope="col">Chosen parameters</th>
            <th scope="col" className="num">
              IS trades
            </th>
            <th scope="col" className="num">
              IS net
            </th>
            <th scope="col" className="num">
              OOS trades
            </th>
            <th scope="col" className="num">
              OOS net
            </th>
            <th scope="col" className="num">
              Equity out
            </th>
          </tr>
        </thead>
        <tbody>
          {(report.windows ?? []).map((window) => (
            <tr key={window.index} className={window.skipped ? "flagged" : undefined}>
              <td className="num">{window.index + 1}</td>
              <td>
                {utcDate(window.train_start)} → {utcDate(window.train_end)}
              </td>
              <td>
                {utcDate(window.test_start)} → {utcDate(window.test_end)}
              </td>
              <td>
                {window.skipped ? (
                  <span style={{ whiteSpace: "normal", color: "var(--warn)" }}>
                    discarded — {window.skip_reason}
                  </span>
                ) : Object.keys(window.chosen_params ?? {}).length ? (
                  <span className="mono">
                    {Object.entries(window.chosen_params ?? {})
                      .map(([key, value]) => `${shortParam(key)}=${String(value)}`)
                      .join(" ")}
                  </span>
                ) : (
                  <span style={{ color: "var(--ink-faint)" }}>fixed spec</span>
                )}
              </td>
              <td className="num">{window.skipped ? "—" : int(window.train_trades)}</td>
              <td className="num">
                <Signed
                  value={window.train_metrics?.net_pnl ?? null}
                  text={signedMoney(window.train_metrics?.net_pnl, 2)}
                />
              </td>
              <td className="num">{window.skipped ? "—" : int(window.test_trades)}</td>
              <td className="num">
                <strong>
                  <Signed
                    value={window.test_metrics?.net_pnl ?? null}
                    text={signedMoney(window.test_metrics?.net_pnl, 2)}
                  />
                </strong>
              </td>
              <td className="num">{money(window.equity_end)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DegradationTable({ report }: { report: WalkForwardResponse }) {
  if ((report.degradation ?? []).length === 0) return null;
  return (
    <div className="table-scroll">
      <table>
        <caption>
          In-sample against out-of-sample, averaged over the usable windows. The standard
          error says how much of the gap is measurement noise.
        </caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col" className="num">
              Windows
            </th>
            <th scope="col" className="num">
              IS mean
            </th>
            <th scope="col" className="num">
              ± se
            </th>
            <th scope="col" className="num">
              OOS mean
            </th>
            <th scope="col" className="num">
              ± se
            </th>
            <th scope="col" className="num">
              OOS / IS
            </th>
          </tr>
        </thead>
        <tbody>
          {(report.degradation ?? []).map((row) => (
            <tr key={row.metric}>
              <th scope="row" style={{ fontWeight: 400 }}>
                {row.metric}
              </th>
              <td className="num">{int(row.observations)}</td>
              <td className="num">{num(row.in_sample_mean, 4)}</td>
              <td className="num" style={{ color: "var(--ink-soft)" }}>
                {row.in_sample_stderr === null ? "—" : num(row.in_sample_stderr, 4)}
              </td>
              <td className="num">{num(row.out_of_sample_mean, 4)}</td>
              <td className="num" style={{ color: "var(--ink-soft)" }}>
                {row.out_of_sample_stderr === null ? "—" : num(row.out_of_sample_stderr, 4)}
              </td>
              <td className="num">
                {row.ratio === null || row.ratio === undefined ? (
                  <span
                    style={{ color: "var(--ink-faint)", whiteSpace: "normal", fontSize: 11 }}
                  >
                    {row.ratio_note}
                  </span>
                ) : (
                  pct(row.ratio)
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StabilityTable({ report }: { report: WalkForwardResponse }) {
  if ((report.parameter_stability ?? []).length === 0) return null;
  return (
    <div className="table-scroll">
      <table>
        <caption>
          Which value the optimizer picked, window after window. A parameter that changes
          every time is the optimizer following noise, and it will not survive forward.
        </caption>
        <thead>
          <tr>
            <th scope="col">Parameter</th>
            <th scope="col" className="num">
              Windows
            </th>
            <th scope="col" className="num">
              Distinct values
            </th>
            <th scope="col">Sequence</th>
            <th scope="col">Most frequent</th>
            <th scope="col" className="num">
              Agreement
            </th>
          </tr>
        </thead>
        <tbody>
          {(report.parameter_stability ?? []).map((row) => (
            <tr key={row.parameter}>
              <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                {row.parameter}
              </th>
              <td className="num">{int(row.observations)}</td>
              <td className="num">{int(row.distinct_values)}</td>
              <td className="mono">{row.chosen.join(" → ")}</td>
              <td className="mono">{row.mode}</td>
              <td className="num">
                {pct(row.mode_share)}{" "}
                {row.mode_share < 0.5 && <Badge kind="warn">unstable</Badge>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// -- permutation ----------------------------------------------------------

function PermutationSection({ runId }: { runId: string }) {
  const [iterations, setIterations] = useState(1000);
  const [report, setReport] = useState<PermutationResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      setReport(await api.permutation({ run_id: runId, iterations }));
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      title="Permutation tests"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          both nulls run through the same engine as the real strategy
        </span>
      }
    >
      <div className="stack">
        <div className="form-grid">
          <Field
            label="Iterations"
            htmlFor="perm-n"
            hint="per test; 1000 takes about a minute"
          >
            <NumberField
              id="perm-n"
              min={10}
              max={20000}
              step={100}
              value={iterations}
              onChange={(value) => setIterations(value ?? 10)}
            />
          </Field>
        </div>
        <div className="actions">
          <button className="primary" onClick={run} disabled={busy}>
            {busy ? "Running the null models…" : "Run permutation tests"}
          </button>
        </div>

        {error !== null && <ErrorNotice error={error} />}
        {busy && (
          <Loading
            label={`Simulating ${int(iterations)} draws per null model. This runs the full engine every time.`}
            height={220}
          />
        )}

        {(report?.tests ?? []).map((test) => (
          <PermutationBlock key={test.kind} test={test} />
        ))}
      </div>
    </Panel>
  );
}

function PermutationBlock({ test }: { test: PermutationTest }) {
  const statistics = test.statistics ?? [];
  const best = statistics.reduce<number>(
    (lowest, statistic) => Math.min(lowest, statistic.p_value),
    1,
  );
  const bars = useMemo(() => {
    if (!test.histogram) return [];
    const { edges, counts } = test.histogram;
    return counts.map((count, index) => ({
      centre: ((edges[index] ?? 0) + (edges[index + 1] ?? 0)) / 2,
      count,
    }));
  }, [test]);

  return (
    <div className="stack">
      <h3>
        {test.kind === "random_entries" ? "Random entries" : "Permuted returns (block bootstrap)"}
      </h3>
      <p className="footnote" style={{ maxWidth: "72ch" }}>
        {test.description}
      </p>

      <Notice kind={best <= 0.05 ? "ok" : "warn"} title="Result against the null">
        {test.verdict}
      </Notice>
      {(test.warnings ?? []).map((warning) => (
        <Notice key={warning} kind="warn" title="Careful">
          {warning}
        </Notice>
      ))}

      <dl className="facts">
        <dt>Iterations</dt>
        <dd>
          {int(test.iterations)} in {num(test.elapsed_seconds, 1)} s
        </dd>
        <dt>Trades: real vs null</dt>
        <dd>
          {int(test.observed_trades)} vs {num(test.null_trades_mean, 1)} ±{" "}
          {num(test.null_trades_std, 1)}
        </dd>
        {test.block_bars !== null && test.block_bars !== undefined && (
          <>
            <dt>Block length</dt>
            <dd>{int(test.block_bars)} bars</dd>
          </>
        )}
      </dl>

      {bars.length > 0 && test.histogram && (
        <div>
          <div className="legend" style={{ padding: "4px 0" }}>
            <span>
              <span className="swatch" style={{ background: NULL_COLOR }} />
              Null distribution of net PnL
            </span>
            <span>
              <span className="swatch" style={{ background: OBSERVED_COLOR }} />
              The real strategy
            </span>
          </div>
          <div className="chart-frame" style={{ height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={bars} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                <CartesianGrid stroke="#eceff1" vertical={false} />
                <XAxis
                  dataKey="centre"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={{ fontSize: 11 }}
                  tickFormatter={(value: number) => num(value, 0)}
                />
                <YAxis tick={{ fontSize: 11 }} width={44} />
                <Tooltip
                  isAnimationActive={false}
                  cursor={{ fill: "rgba(0,0,0,0.04)" }}
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const row = payload[0]?.payload as { centre: number; count: number };
                    return (
                      <div className="tooltip">
                        <div className="t">net PnL around {num(row.centre, 2)}</div>
                        <dl>
                          <dt>draws</dt>
                          <dd>{int(row.count)}</dd>
                        </dl>
                      </div>
                    );
                  }}
                />
                <ReferenceLine
                  x={test.histogram.observed}
                  stroke={OBSERVED_COLOR}
                  strokeWidth={2}
                  label={{
                    value: "real",
                    position: "top",
                    fontSize: 11,
                    fill: OBSERVED_COLOR,
                  }}
                />
                <Bar dataKey="count" isAnimationActive={false}>
                  {bars.map((_, index) => (
                    <Cell key={index} fill={NULL_COLOR} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <div className="table-scroll">
        <table>
          <caption>
            The p-value is (1 + draws at or above the real result) / (1 + iterations): it can
            never reach zero, however many draws are taken.
          </caption>
          <thead>
            <tr>
              <th scope="col">Statistic</th>
              <th scope="col" className="num">
                Real
              </th>
              <th scope="col" className="num">
                Null p05
              </th>
              <th scope="col" className="num">
                Null median
              </th>
              <th scope="col" className="num">
                Null p95
              </th>
              <th scope="col" className="num">
                Percentile
              </th>
              <th scope="col" className="num">
                p-value
              </th>
            </tr>
          </thead>
          <tbody>
            {statistics.map((statistic) => (
              <tr key={statistic.key}>
                <th scope="row" style={{ fontWeight: 400 }}>
                  {statistic.label}
                </th>
                <td className="num">
                  <strong>
                    <Signed value={statistic.observed} text={num(statistic.observed, 4)} />
                  </strong>
                </td>
                <td className="num">{num(statistic.null_p05, 4)}</td>
                <td className="num">{num(statistic.null_p50, 4)}</td>
                <td className="num">{num(statistic.null_p95, 4)}</td>
                <td className="num">{num(statistic.percentile, 1)}</td>
                <td className="num">
                  {num(statistic.p_value, 4)}{" "}
                  {statistic.p_value <= 0.05 ? (
                    <Badge kind="ok">beats null</Badge>
                  ) : (
                    <Badge kind="mute">inside null</Badge>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// -- multiple testing -----------------------------------------------------

function MultipleTestingSection({ runId }: { runId: string }) {
  const [report, setReport] = useState<MultipleTestingResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    setBusy(true);
    setError(null);
    setReport(null);
    api
      .multipleTesting({ run_id: runId })
      .then(setReport)
      .catch(setError)
      .finally(() => setBusy(false));
  }, [runId]);

  const dsr = report?.deflated_sharpe;

  return (
    <Panel
      title="Multiple testing"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          trials counted from the run store, not typed in by hand
        </span>
      }
    >
      {error !== null && <ErrorNotice error={error} />}
      {busy && <Loading label="Counting the trials on this data…" height={140} />}
      {report && (
        <div className="stack">
          {(report.warnings ?? []).map((warning) => (
            <Notice key={warning} kind="warn" title="Read this first">
              {warning}
            </Notice>
          ))}

          <Notice kind="info" title="What the search costs">
            {report.verdict}
          </Notice>

          {dsr && (
            <div className="pair">
              <div>
                <div className="label">Observed Sharpe</div>
                <div className="value">
                  <Signed
                    value={dsr.observed_sharpe_per_trade}
                    text={num(dsr.observed_sharpe_per_trade, 4)}
                  />
                </div>
                <div className="sub">
                  per trade, over {int(dsr.observations)} trades
                  {dsr.observed_sharpe_annualized !== null &&
                    dsr.observed_sharpe_annualized !== undefined &&
                    ` · ${num(dsr.observed_sharpe_annualized, 2)} annualized`}
                </div>
              </div>
              <div>
                <div className="label">Deflated Sharpe</div>
                <div className="value">
                  {dsr.valid ? pct(dsr.deflated_sharpe) : "—"}
                </div>
                <div className="sub">
                  {dsr.valid
                    ? `probability the true Sharpe is above zero after ${int(dsr.trials)} trials`
                    : dsr.reason}
                </div>
              </div>
            </div>
          )}

          {dsr?.valid && (
            <dl className="facts">
              <dt>Sharpe the search buys for free</dt>
              <dd>{num(dsr.expected_max_sharpe, 4)} (expected max of {int(dsr.trials)} trials)</dd>
              <dt>Probabilistic Sharpe (uncorrected)</dt>
              <dd>{pct(dsr.probabilistic_sharpe)}</dd>
              <dt>Skew / kurtosis of trade PnL</dt>
              <dd>
                {num(dsr.skewness, 3)} / {num(dsr.kurtosis, 3)}
              </dd>
              <dt>Sharpe variance across trials</dt>
              <dd>{num(dsr.variance_across_trials, 6)}</dd>
            </dl>
          )}

          {(report.thresholds ?? []).length > 0 && (
            <div className="table-scroll">
              <table>
                <caption>
                  The family of trials on this instrument and this data, ranked by p-value.
                  Both corrections are shown: they disagree exactly when the answer is
                  delicate. A negative mean PnL turns a small p-value into evidence of
                  reliable losing.
                </caption>
                <thead>
                  <tr>
                    <th scope="col" className="num">
                      Rank
                    </th>
                    <th scope="col">Run</th>
                    <th scope="col">Strategy</th>
                    <th scope="col" className="num">
                      p-value
                    </th>
                    <th scope="col" className="num">
                      Bonferroni
                    </th>
                    <th scope="col" className="num">
                      B-H
                    </th>
                    <th scope="col">Mean PnL</th>
                    <th scope="col">Survives</th>
                  </tr>
                </thead>
                <tbody>
                  {(report.thresholds ?? []).map((row) => (
                    <tr key={row.run_id}>
                      <td className="num">{row.rank}</td>
                      <td className="mono">
                        <a href={`#/result/${row.run_id}`}>{row.run_id.slice(0, 8)}</a>
                      </td>
                      <td>{row.strategy_id}</td>
                      <td className="num">{num(row.p_value, 4)}</td>
                      <td className="num">{num(row.bonferroni_threshold, 4)}</td>
                      <td className="num">{num(row.benjamini_hochberg_threshold, 4)}</td>
                      <td>
                        {row.mean_pnl_positive ? (
                          <Badge kind="ok">+ positive</Badge>
                        ) : (
                          <Badge kind="bad">− negative</Badge>
                        )}
                      </td>
                      <td>
                        {row.passes_bonferroni && <Badge kind="ok">Bonferroni</Badge>}{" "}
                        {row.passes_benjamini_hochberg && <Badge kind="info">B-H</Badge>}
                        {!row.passes_bonferroni && !row.passes_benjamini_hochberg && (
                          <Badge kind="mute">neither</Badge>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div>
            <h3>Probability of backtest overfitting</h3>
            {report.pbo.valid ? (
              <dl className="facts">
                <dt>PBO</dt>
                <dd>
                  <strong>{pct(report.pbo.pbo)}</strong> over {int(report.pbo.splits)} splits
                  of {int(report.pbo.slices)} slices, {int(report.pbo.candidates)} candidates
                </dd>
              </dl>
            ) : (
              <Notice kind="info" title="Not computed">
                {report.pbo.reason}
              </Notice>
            )}
          </div>
        </div>
      )}
    </Panel>
  );
}

// -- tick resolve ---------------------------------------------------------

function TickResolveSection({ runId }: { runId: string }) {
  const [report, setReport] = useState<TickResolveResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function run() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      setReport(await api.tickResolve({ run_id: runId }));
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel
      title="Ambiguous trades, settled with ticks"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          needs MetaTrader 5 connected; without it nothing is assumed
        </span>
      }
    >
      <div className="stack">
        <p className="footnote" style={{ maxWidth: "72ch" }}>
          When one bar touches both the stop and the target, the engine assumes the stop. That
          is conservative, and it is still an assumption. Ticks say which level came first.
        </p>
        <div className="actions">
          <button onClick={run} disabled={busy}>
            {busy ? "Reading the ticks…" : "Resolve with tick data"}
          </button>
        </div>

        {error !== null && <ErrorNotice error={error} />}
        {busy && <Loading label="Downloading the ticks of each ambiguous bar…" height={140} />}

        {report && (
          <>
            <Notice kind={report.available ? "info" : "warn"} title="Resolution">
              {report.verdict}
            </Notice>
            {(report.warnings ?? []).map((warning) => (
              <Notice key={warning} kind="warn" title="Careful">
                {warning}
              </Notice>
            ))}

            <dl className="facts">
              <dt>Ambiguous trades</dt>
              <dd>{int(report.ambiguous_trades)}</dd>
              <dt>Resolved / unresolved</dt>
              <dd>
                {int(report.resolved)} / {int(report.unresolved)}
              </dd>
              <dt>Actually reached the target first</dt>
              <dd>{int(report.resolved_to_take_profit)}</dd>
              <dt>Net PnL: assumed → resolved</dt>
              <dd>
                <Signed
                  value={report.original_net_pnl}
                  text={signedMoney(report.original_net_pnl)}
                />
                {" → "}
                <Signed
                  value={report.resolved_net_pnl}
                  text={signedMoney(report.resolved_net_pnl)}
                />{" "}
                (<Signed value={report.delta_net_pnl} text={signedMoney(report.delta_net_pnl)} />
                )
              </dd>
              <dt>Win rate: assumed → resolved</dt>
              <dd>
                {pct(report.original_win_rate)} → {pct(report.resolved_win_rate)}
              </dd>
            </dl>

            {(report.trades ?? []).length > 0 && (
              <div className="table-scroll" style={{ maxHeight: 320, overflowY: "auto" }}>
                <table>
                  <caption>
                    Every ambiguous trade and what the ticks said. A long is read on the bid, a
                    short on the ask, exactly as the engine prices the exit.
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col" className="num">
                        #
                      </th>
                      <th scope="col">Dir</th>
                      <th scope="col">Exit bar (UTC)</th>
                      <th scope="col" className="num">
                        Ticks
                      </th>
                      <th scope="col">First touch</th>
                      <th scope="col" className="num">
                        Assumed
                      </th>
                      <th scope="col" className="num">
                        Resolved
                      </th>
                      <th scope="col" className="num">
                        Δ
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {(report.trades ?? []).map((trade) => (
                      <tr
                        key={trade.index}
                        className={trade.resolution === "take_profit" ? "flagged" : undefined}
                      >
                        <td className="num">{trade.index + 1}</td>
                        <td>{trade.direction > 0 ? "LONG" : "SHORT"}</td>
                        <td>{utcDateTime(trade.exit_time)}</td>
                        <td className="num">{int(trade.ticks)}</td>
                        <td>
                          {trade.resolution === "take_profit" && (
                            <Badge kind="ok">take profit</Badge>
                          )}
                          {trade.resolution === "stop_loss" && (
                            <Badge kind="bad">stop loss</Badge>
                          )}
                          {trade.resolution === "unresolved" && (
                            <Badge kind="mute">unresolved</Badge>
                          )}
                        </td>
                        <td className="num">{signedMoney(trade.original_net_pnl, 3)}</td>
                        <td className="num">{signedMoney(trade.resolved_net_pnl, 3)}</td>
                        <td className="num">
                          <Signed value={trade.delta} text={signedMoney(trade.delta, 3)} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}
