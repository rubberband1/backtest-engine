import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type CompareResponse, type RunSummary } from "../api/client";
import { Badge, Empty, ErrorNotice, Loading, Notice, Panel, Signed } from "../components/ui";
import { byFormat, int, num, pct, utcDate, utcDateTime } from "../format";
import { DRAW_MS, useFirstDraw } from "../motion";

/** Colours that stay distinguishable printed in greyscale (different luminance). */
const SERIES_COLORS = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
];

export function ComparePage({ initialRunIds }: { initialRunIds: string[] }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [selected, setSelected] = useState<string[]>(initialRunIds);
  const [comparison, setComparison] = useState<CompareResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [listError, setListError] = useState<unknown>(null);

  useEffect(() => {
    api
      .runs({ limit: 100 })
      .then((all) => setRuns(all.filter((run) => run.status === "done")))
      .catch(setListError);
  }, []);

  useEffect(() => {
    const query = selected.length ? `?runs=${selected.join(",")}` : "";
    window.history.replaceState(null, "", `#/compare${query}`);

    if (selected.length < 2) {
      setComparison(null);
      setError(null);
      return;
    }
    let cancelled = false;
    setBusy(true);
    setError(null);
    api
      .compare(selected)
      .then((data) => !cancelled && setComparison(data))
      .catch((problem) => !cancelled && setError(problem))
      .finally(() => !cancelled && setBusy(false));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  function toggle(runId: string) {
    setSelected((current) =>
      current.includes(runId)
        ? current.filter((item) => item !== runId)
        : [...current, runId],
    );
  }

  if (listError !== null) return <ErrorNotice error={listError} />;
  if (!runs) return <Loading label="Loading the runs…" height={220} />;

  return (
    <div className="row">
      <div className="grow" style={{ flexBasis: 340, maxWidth: 420 }}>
        <Panel
          title="Runs to compare"
          aside={<span style={{ fontSize: 12 }}>{selected.length} selected</span>}
          tight
        >
          {runs.length === 0 ? (
            <Empty title="No completed run">
              <p>
                <a href="#/">Launch a backtest</a> to have something to compare.
              </p>
            </Empty>
          ) : (
            <div className="checklist">
              {runs.map((run) => {
                const index = selected.indexOf(run.run_id);
                return (
                  <label key={run.run_id}>
                    <input
                      type="checkbox"
                      checked={index >= 0}
                      onChange={() => toggle(run.run_id)}
                    />
                    <span
                      className="swatch"
                      style={{
                        background:
                          index >= 0
                            ? SERIES_COLORS[index % SERIES_COLORS.length]
                            : "transparent",
                        border: "1px solid var(--line-strong)",
                      }}
                      aria-hidden="true"
                    />
                    <span>
                      <span className="mono">{run.run_id.slice(0, 8)}</span> · {run.strategy_id}
                      <br />
                      <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
                        {run.symbol} {run.timeframe} · {utcDate(run.data_start)} →{" "}
                        {utcDate(run.data_end)}
                      </span>
                    </span>
                    <span className="meta">
                      <Signed
                        value={run.total_return}
                        text={run.total_return === null ? "—" : pct(run.total_return)}
                      />
                    </span>
                  </label>
                );
              })}
            </div>
          )}
          <div className="pager">
            <span>At least two runs are needed. The selection order fixes the reference.</span>
          </div>
        </Panel>
      </div>

      <div className="grow" style={{ flexBasis: 640 }}>
        {error !== null && <ErrorNotice error={error} />}
        {!error && selected.length < 2 && (
          <Panel title="Comparison">
            <Empty title="Select at least two runs">
              <p>
                The curves are normalized to 100 at the start of their own period, so two runs
                with different initial equity or different periods stay comparable.
              </p>
            </Empty>
          </Panel>
        )}
        {busy && selected.length >= 2 && (
          <Panel title="Comparison">
            <Loading label="Aligning the curves…" height={300} />
          </Panel>
        )}
        {!busy && comparison && <ComparisonView comparison={comparison} />}
      </div>
    </div>
  );
}

function ComparisonView({ comparison }: { comparison: CompareResponse }) {
  // Keyed on the set being compared, so picking a different set of runs draws
  // the new curves once and changing a filter on the same set does not.
  const draw = useFirstDraw(comparison.runs.map((run) => run.run_id).join(","));
  const data = useMemo(
    () =>
      comparison.series.map((point) => {
        const row: Record<string, number | null> = { ts: new Date(point.t).getTime() };
        point.values.forEach((value, index) => {
          row[`v${index}`] = value;
        });
        return row;
      }),
    [comparison],
  );

  const configDiff = comparison.config_diff ?? [];
  const specDiff = comparison.symbol_spec_diff ?? [];

  /**
   * What separates two columns, in one line under the run id. Without it a
   * table of near-identical numbers gives no clue why they differ at all.
   */
  function diffSummary(index: number): string | null {
    if (configDiff.length === 0) return null;
    return configDiff
      .map((row) => `${row.key} ${row.values[index] ?? "—"}`)
      .join(" · ");
  }

  return (
    <div className="stack">
      {(comparison.warnings ?? []).map((warning) => (
        <Notice key={warning} kind="warn" title="Careful with this comparison">
          {warning}
        </Notice>
      ))}

      <Panel
        title="Equity curves normalized to 100"
        aside={
          <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            {utcDate(comparison.axis_start)} → {utcDate(comparison.axis_end)} UTC ·{" "}
            {int(comparison.series.length)} points
          </span>
        }
        tight
      >
        <div className="legend">
          {comparison.runs.map((run, index) => (
            <span key={run.run_id}>
              <span
                className="swatch"
                style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
              />
              {run.label}
            </span>
          ))}
        </div>
        <div className="chart-frame" style={{ height: 340, padding: "8px 8px 0" }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
              <CartesianGrid stroke="var(--chart-grid)" />
              <XAxis
                dataKey="ts"
                type="number"
                scale="time"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(value: number) => utcDate(new Date(value).toISOString())}
                tick={{ fontSize: 11 }}
                minTickGap={56}
              />
              <YAxis
                tick={{ fontSize: 11 }}
                width={56}
                domain={["auto", "auto"]}
                tickFormatter={(value: number) => num(value, 0)}
              />
              <ReferenceLine y={100} stroke="var(--chart-axis)" strokeDasharray="4 3" />
              <Tooltip
                isAnimationActive={false}
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null;
                  return (
                    <div className="tooltip">
                      <div className="t">
                        {utcDateTime(new Date((label as number) ?? 0).toISOString())} UTC
                      </div>
                      <dl>
                        {payload.map((entry) => {
                          const index = Number(String(entry.dataKey).slice(1));
                          const run = comparison.runs[index];
                          return (
                            <TooltipRow
                              key={String(entry.dataKey)}
                              name={run ? run.run_id.slice(0, 8) : String(entry.dataKey)}
                              value={entry.value as number | null}
                            />
                          );
                        })}
                      </dl>
                    </div>
                  );
                }}
              />
              {comparison.runs.map((run, index) => (
                <Line
                  key={run.run_id}
                  type="monotone"
                  dataKey={`v${index}`}
                  name={run.label}
                  stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                  strokeWidth={1.6}
                  dot={false}
                  connectNulls={false}
                  isAnimationActive={draw}
                  animationDuration={DRAW_MS}
                  animationEasing="ease-out"
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel
        title="Metrics side by side"
        aside={
          comparison.configs_identical ? (
            <Badge kind="info">identical configurations</Badge>
          ) : (
            <Badge kind="warn">
              {configDiff.length} differing field{configDiff.length === 1 ? "" : "s"}
            </Badge>
          )
        }
        tight
      >
        {comparison.configs_identical && (
          <div style={{ padding: "16px 16px 0" }}>
            <Notice kind="info" title="The configurations are identical">
              No config field and no strategy id differs between these runs: any difference in
              the numbers comes from the data window, not from the setup.
            </Notice>
          </div>
        )}
        <div className="table-scroll">
          <table>
            <caption>
              The Δ column is the difference against the first selected run. The arrow says
              whether the difference goes the right way for that metric.
            </caption>
            <thead>
              <tr>
                <th scope="col">Metric</th>
                {comparison.runs.map((run, index) => {
                  const summary = diffSummary(index);
                  return (
                    <th key={run.run_id} className="num" scope="col">
                      <span
                        className="swatch"
                        style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
                      />
                      <span className="mono">{run.run_id.slice(0, 8)}</span>
                      {index === 0 && " (ref.)"}
                      {summary && <span className="col-sub">{summary}</span>}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {comparison.metrics.map((row) => (
                <tr key={row.key}>
                  <th scope="row" style={{ fontWeight: 400 }}>
                    {row.label}
                    {row.higher_is_better !== null && (
                      <span style={{ color: "var(--ink-faint)" }}>
                        {row.higher_is_better ? " ↑ better" : " ↓ better"}
                      </span>
                    )}
                  </th>
                  {row.values.map((value, index) => {
                    const delta = row.delta[index] ?? null;
                    const better =
                      row.higher_is_better === null || delta === null || delta === 0
                        ? null
                        : row.higher_is_better === delta > 0;
                    return (
                      <td className="num" key={`${row.key}-${index}`}>
                        {byFormat(value, row.format)}
                        {index > 0 && delta !== null && delta !== 0 && (
                          <span
                            className={better === null ? "flat" : better ? "pos" : "neg"}
                            style={{ marginLeft: 6, fontSize: 12 }}
                          >
                            {better === null ? "" : better ? "▲" : "▼"}{" "}
                            {byFormat(delta, row.format, true)}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}

              {configDiff.length > 0 && (
                <>
                  <tr>
                    <th
                      scope="row"
                      colSpan={comparison.runs.length + 1}
                      className="section-row"
                    >
                      Configuration differences
                    </th>
                  </tr>
                  {configDiff.map((row) => (
                    <tr key={`config-${row.key}`}>
                      <th scope="row" style={{ fontWeight: 400 }}>
                        {row.key}
                      </th>
                      {row.values.map((value, index) => (
                        <td className="num mono" key={`${row.key}-${index}`}>
                          {value ?? "—"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </>
              )}

              {/* The instrument spec is an input to the result, not context:
                  tick_value tracks an FX rate and swap rates move when the
                  broker decides. Two runs differing here were priced
                  differently, whatever else matches. */}
              {specDiff.length > 0 && (
                <>
                  <tr>
                    <th
                      scope="row"
                      colSpan={comparison.runs.length + 1}
                      className="section-row"
                    >
                      Instrument specification differences — these change what a trade costs
                    </th>
                  </tr>
                  {specDiff.map((row) => (
                    <tr key={`spec-${row.key}`} className="flagged">
                      <th scope="row" style={{ fontWeight: 400 }}>
                        {row.key}
                      </th>
                      {row.values.map((value, index) => (
                        <td className="num mono" key={`${row.key}-${index}`}>
                          {value ?? "—"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </>
              )}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title="Runs compared" tight>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Strategy</th>
                <th scope="col">Instrument</th>
                <th scope="col">Period (UTC)</th>
                <th scope="col" className="num">
                  Initial equity
                </th>
                <th scope="col" />
              </tr>
            </thead>
            <tbody>
              {comparison.runs.map((run, index) => (
                <tr key={run.run_id}>
                  <td className="mono">
                    <span
                      className="swatch"
                      style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
                    />
                    {run.run_id.slice(0, 8)}
                  </td>
                  <td>{run.strategy_id}</td>
                  <td>
                    {run.symbol} {run.timeframe}
                  </td>
                  <td>
                    {utcDate(run.start)} → {utcDate(run.end)}
                  </td>
                  <td className="num">{num(run.initial_equity, 2)}</td>
                  <td>
                    <a href={`#/result/${run.run_id}`}>detail →</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </div>
  );
}

function TooltipRow({ name, value }: { name: string; value: number | null }) {
  return (
    <>
      <dt className="mono">{name}</dt>
      <dd>{value === null ? "—" : num(value, 2)}</dd>
    </>
  );
}
