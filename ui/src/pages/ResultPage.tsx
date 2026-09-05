import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  type Breakeven,
  type Equity,
  type Gates,
  type Metrics,
  type RunDetail,
  type Trades,
  type Uncertainty,
} from "../api/client";
import {
  Badge,
  Empty,
  ErrorNotice,
  Loading,
  Notice,
  Panel,
  Signed,
  StatusBadge,
} from "../components/ui";
import {
  duration,
  int,
  money,
  num,
  pct,
  signedMoney,
  signedPct,
  utcDate,
  utcDateTime,
} from "../format";
import { DRAW_MS, useFirstDraw } from "../motion";

const PAGE_SIZE = 25;
const EQUITY_COLOR = "var(--chart-series)";
const DRAWDOWN_COLOR = "var(--chart-adverse)";

type ChartPoint = { ts: number; equity: number; drawdown: number };

export function ResultPage({ runId }: { runId: string }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [equity, setEquity] = useState<Equity | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    if (!runId) return;
    setRun(null);
    setEquity(null);
    setError(null);
    api
      .run(runId)
      .then((detail) => {
        setRun(detail);
        if (detail.status === "done") return api.equity(runId).then(setEquity);
        return undefined;
      })
      .catch(setError);
  }, [runId]);

  if (!runId) {
    return (
      <Panel title="No run selected">
        <Empty title="Open a run from the Run page">
          <p>
            <a href="#/">Go to the configuration</a> and launch a backtest, or pick one of the
            recent runs.
          </p>
        </Empty>
      </Panel>
    );
  }
  if (error !== null) return <ErrorNotice error={error} />;
  if (!run) return <Loading label="Loading the run…" height={220} />;

  if (run.status !== "done") {
    return (
      <Panel title={`Run ${runId.slice(0, 8)}`} aside={<StatusBadge status={run.status} />}>
        {run.status === "running" ? (
          <Loading label="The backtest is still running. Reload in a moment." height={80} />
        ) : (
          <Notice kind="error" title="The run failed">
            {run.error ?? "no detail available"}
          </Notice>
        )}
      </Panel>
    );
  }

  const strategy = run.strategy;
  const benchmark = run.benchmark;

  return (
    <div className="stack">
      <RunHeader run={run} />
      {run.uncertainty && <UncertaintyPanel band={run.uncertainty} runId={runId} />}
      {strategy && (
        <EquitySection equity={equity} initial={strategy.initial_equity} runId={runId} />
      )}
      <div className="row">
        <div className="grow" style={{ flexBasis: 560 }}>
          {strategy && <MetricsTable strategy={strategy} benchmark={benchmark ?? null} />}
        </div>
        <div className="grow" style={{ flexBasis: 380 }}>
          <div className="stack">
            {run.breakeven && (
              <BreakevenPanel breakeven={run.breakeven} realized={strategy?.win_rate ?? null} />
            )}
            <ExecutionPanel run={run} />
            {run.spread_coverage && (
              <SpreadCoveragePanel coverage={run.spread_coverage} />
            )}
            {run.gates && <GatesPanel gates={run.gates} />}
            <SymbolSpecPanel run={run} />
          </div>
        </div>
      </div>
      <TradesPanel runId={runId} />
    </div>
  );
}

function SpreadCoveragePanel({
  coverage,
}: {
  coverage: NonNullable<RunDetail["spread_coverage"]>;
}) {
  const share = coverage.measured_share;
  // Not a quality score: a run at 0% is not wrong, it is resting on a cost
  // taken from another period. The label says which, and never the colour
  // alone.
  const kind = coverage.fully_measured ? "ok" : share > 0 ? "warn" : "bad";
  const label = coverage.fully_measured
    ? "measured throughout"
    : share > 0
      ? "partly assumed"
      : "assumed throughout";

  return (
    <Panel title="Spread: measured or assumed" aside={<Badge kind={kind}>{label}</Badge>}>
      <div className="stack">
        <div className="pair">
          <div>
            <div className="label">Bars with an M1 sample</div>
            <div className="value">{pct(share)}</div>
            <div className="sub">
              {int(coverage.measured_bars)} of {int(coverage.bars)}
            </div>
          </div>
          <div>
            <div className="label">Bars on an assumed spread</div>
            <div className="value">{int(coverage.assumed_bars)}</div>
            <div className="sub">charged a constant from another period</div>
          </div>
        </div>

        <dl className="facts">
          <dt>M1 window</dt>
          <dd>
            {coverage.m1_window_start && coverage.m1_window_end
              ? `${utcDate(coverage.m1_window_start)} → ${utcDate(coverage.m1_window_end)}`
              : "none cached for this period"}
          </dd>
        </dl>

        <p className="footnote">
          Above M1 the broker's spread field is the minimum of the M1 spreads
          inside the bar, so it is never charged. Where M1 exists the spread is
          rebuilt from it; everywhere else a measured constant is applied to
          bars it was not measured on. That is an assumption, and on the oldest
          part of a sample it is the assumption most able to move a marginal
          result.
        </p>
      </div>
    </Panel>
  );
}

function RunHeader({ run }: { run: RunDetail }) {
  const strategy = run.strategy;
  const config = run.config as Record<string, unknown>;
  return (
    <Panel
      className="reveal"
      title={
        <div>
          <h1>
            {run.spec.name as string} <span style={{ color: "var(--ink-faint)" }}>·</span>{" "}
            {String(config.symbol)} {String(config.timeframe)}
          </h1>
          <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            {utcDate(run.data_start)} → {utcDate(run.data_end)} UTC · {int(run.bars)} bars ·
            computed in {duration(run.duration_seconds)}
          </div>
        </div>
      }
      aside={
        <div style={{ textAlign: "right", fontSize: 12, color: "var(--ink-faint)" }}>
          <StatusBadge status={run.status} /> <span className="mono">{run.run_id}</span>
          <div>
            engine {run.engine_version} · spec {run.spec_hash.slice(0, 8)} · data{" "}
            {run.data_hash.slice(0, 8)}
          </div>
          <div style={{ marginTop: 4 }}>
            <a href={`#/validation/${run.run_id}`}>validate this run →</a>
          </div>
        </div>
      }
      tight
    >
      {strategy && (
        <div className="kpis">
          <Kpi
            label="Final equity"
            value={money(strategy.final_equity)}
            sub={`from ${money(strategy.initial_equity)}`}
          />
          <Kpi
            label="Return"
            value={pct(strategy.total_return)}
            tone={strategy.total_return}
            sub={`annual ${pct(strategy.annual_return)}`}
          />
          <Kpi
            label="Max drawdown"
            value={pct(strategy.max_drawdown_pct)}
            sub={`${money(strategy.max_drawdown_money)} in currency`}
          />
          <Kpi
            label="Trades"
            value={int(strategy.trades)}
            sub={`win ${pct(strategy.win_rate)}`}
          />
          <Kpi
            label="Profit factor"
            value={num(strategy.profit_factor, 3)}
            small
            sub={`expectancy ${signedMoney(strategy.expectancy, 4)}`}
          />
          <Kpi
            label="Sharpe / Sortino"
            value={`${num(strategy.sharpe, 2)} / ${num(strategy.sortino, 2)}`}
            small
            sub={`t ${num(strategy.t_stat, 2)} · p ${num(strategy.p_value, 4)}`}
          />
        </div>
      )}
    </Panel>
  );
}

/**
 * The result as a band, not as a number.
 *
 * When a bar touches stop and target together the engine assumes the stop.
 * That assumption is conservative and it is still an assumption: on a run
 * where it decides the sign of the result, quoting a single equity figure is
 * a claim the data does not support. The band is the distance between
 * assuming every ambiguous trade lost and assuming every one of them won —
 * the tick resolution, when it can be run, lands somewhere inside it.
 */
function UncertaintyPanel({ band, runId }: { band: Uncertainty; runId: string }) {
  const wide = band.exceeds_threshold;
  if (band.ambiguous_trades === 0) {
    return (
      <Panel
        title="Uncertainty band"
        aside={<Badge kind="ok">no ambiguous trade</Badge>}
        tight
      >
        <div style={{ padding: "var(--space-2) var(--space-3)", color: "var(--ink-soft)" }}>
          {band.verdict}
        </div>
      </Panel>
    );
  }
  return (
    <Panel
      title="Uncertainty band"
      aside={
        wide ? (
          <Badge kind="bad">not conclusive at bar resolution</Badge>
        ) : (
          <Badge kind="warn">
            {pct(band.ambiguous_share, 1)} ambiguous
          </Badge>
        )
      }
    >
      <div className="stack">
        <Notice kind={wide ? "warn" : "info"} title="What the assumption is worth">
          {band.verdict}
        </Notice>

        <div className="kpis">
          <Kpi
            label="Conservative"
            value={money(band.conservative_final_equity)}
            sub="every ambiguous trade exits on its stop — what the engine reports"
          />
          <Kpi
            label="Optimistic"
            value={money(band.optimistic_final_equity)}
            sub="every ambiguous trade exits on its target — an upper bound, not a result"
          />
          <Kpi
            label="Band"
            value={signedMoney(band.band_money)}
            small
            sub={`${signedPct(band.band_equity_pct)} of initial equity`}
          />
          <Kpi
            label="Ambiguous trades"
            value={`${int(band.ambiguous_trades)} / ${int(band.trades)}`}
            small
            sub={`${pct(band.ambiguous_share, 1)} · threshold ${pct(band.threshold, 0)}`}
          />
          <Kpi
            label="Win rate"
            value={`${pct(band.conservative_win_rate, 1)} – ${pct(band.optimistic_win_rate, 1)}`}
            small
            sub="conservative to optimistic"
          />
        </div>

        {(band.warnings ?? []).map((warning) => (
          <Notice key={warning} kind="warn" title="Careful">
            {warning}
          </Notice>
        ))}

        {wide && (
          <div>
            <a href={`#/validation/${runId}`}>
              Resolve the ambiguous trades against tick data →
            </a>{" "}
            <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
              slow, and the only thing that narrows this band with evidence rather
              than with an assumption.
            </span>
          </div>
        )}
      </div>
    </Panel>
  );
}

/**
 * Signals in, trades out, and every rejection in between.
 *
 * A gate that throws away most of the signals is not a safety margin, it is
 * the strategy: on oil the median spread sat within a point of
 * `max_spread_points` and the gate was silently rejecting about half the
 * entries, which made the run look like a strategy with few opportunities
 * rather than one that was almost never allowed to trade.
 */
function GatesPanel({ gates }: { gates: Gates }) {
  const rows = gates.rows ?? [];
  const warnings = gates.warnings ?? [];
  return (
    <Panel
      title="Risk gates"
      aside={
        warnings.length > 0 ? (
          <Badge kind="warn">{warnings.length} above {pct(gates.threshold, 0)}</Badge>
        ) : (
          <Badge kind="ok">no dominant gate</Badge>
        )
      }
      tight
    >
      <div style={{ padding: "var(--space-2) var(--space-3)" }}>
        <div style={{ color: "var(--ink-soft)", fontSize: 12, marginBottom: 8 }}>
          {gates.verdict}
        </div>
        {rows.length === 0 ? (
          <div style={{ color: "var(--ink-faint)", fontSize: 12 }}>
            Every signal reached execution.
          </div>
        ) : (
          <table>
            <caption>
              Share of the {int(gates.signals)} signals the strategy produced.
            </caption>
            <thead>
              <tr>
                <th scope="col">Gate</th>
                <th scope="col" className="num">
                  Rejected
                </th>
                <th scope="col" className="num">
                  Share
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.code} className={row.exceeds_threshold ? "flagged" : undefined}>
                  <th scope="row" style={{ fontWeight: 400, whiteSpace: "normal" }}>
                    {row.label}
                    {row.exceeds_threshold && (
                      <>
                        {" "}
                        <Badge kind="warn">dominant</Badge>
                      </>
                    )}
                  </th>
                  <td className="num">{int(row.rejected)}</td>
                  <td className="num">{pct(row.share, 1)}</td>
                </tr>
              ))}
              <tr>
                <th scope="row" style={{ fontWeight: 600 }}>
                  Executed
                </th>
                <td className="num">{int(gates.executed)}</td>
                <td className="num">{pct(gates.executed_share, 1)}</td>
              </tr>
            </tbody>
          </table>
        )}
        {warnings.map((warning) => (
          <div key={warning} style={{ marginTop: 8 }}>
            <Notice kind="warn" title="Dominant gate">
              {warning}
            </Notice>
          </div>
        ))}
      </div>
    </Panel>
  );
}

/**
 * The instrument specification this run actually used.
 *
 * tick_value moves with the FX rate and the broker changes swap rates
 * without notice: two identical runs hours apart are not identical, and the
 * run_id covers these fields precisely so that they cannot drift silently.
 * A run persisted before that mechanism existed says so rather than
 * borrowing today's numbers.
 */
function SymbolSpecPanel({ run }: { run: RunDetail }) {
  const spec = run.symbol_spec;
  if (!spec) {
    return (
      <Panel title="Instrument specification" aside={<Badge kind="warn">not registered</Badge>}>
        <Notice kind="warn" title="This run did not pin its SymbolSpec">
          It was persisted before the specification was recorded with the run, so what it
          used cannot be recovered. The values on disk today may differ: the broker moves
          swap rates, and tick_value follows the FX rate. Re-run it to get a run whose
          costs are documented.
        </Notice>
      </Panel>
    );
  }
  return (
    <Panel
      title="Instrument specification"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-faint)" }}>
          read {utcDateTime(run.symbol_spec_read_at)} UTC
        </span>
      }
      tight
    >
      <table>
        <caption>
          The fields in bold are hashed into the run_id: a change in any of them is a
          different run, not the same one with different numbers.
        </caption>
        <tbody>
          <SpecRow label="Point" value={num(spec.point, 5)} hashed />
          <SpecRow label="Digits" value={int(spec.digits)} hashed />
          <SpecRow label="Contract size" value={num(spec.contract_size, 2)} hashed />
          <SpecRow label="Tick value" value={num(spec.tick_value, 6)} hashed />
          <SpecRow label="Tick size" value={num(spec.tick_size, 6)} hashed />
          <SpecRow label="Swap long / short" value={`${num(spec.swap_long, 3)} / ${num(spec.swap_short, 3)}`} hashed />
          <SpecRow
            label="Volume min / step / max"
            value={`${num(spec.volume_min, 2)} / ${num(spec.volume_step, 2)} / ${num(spec.volume_max, 2)}`}
            hashed
          />
          <SpecRow label="Profit currency" value={spec.currency_profit} />
          <SpecRow label="Trade mode" value={spec.trade_mode} />
          <SpecRow
            label="Spec hash"
            value={run.symbol_spec_hash ? run.symbol_spec_hash.slice(0, 16) : "—"}
          />
        </tbody>
      </table>
    </Panel>
  );
}

function SpecRow({ label, value, hashed }: { label: string; value: string; hashed?: boolean }) {
  return (
    <tr>
      <th scope="row" style={{ fontWeight: hashed ? 600 : 400 }}>
        {label}
      </th>
      <td className="num mono">{value}</td>
    </tr>
  );
}

function Kpi(props: {
  label: string;
  value: string;
  sub?: string;
  small?: boolean;
  tone?: number | null;
}) {
  return (
    <div className="kpi">
      <div className="label">{props.label}</div>
      <div className={`value${props.small ? " small" : ""}`}>
        {props.tone === undefined ? (
          props.value
        ) : (
          <Signed value={props.tone} text={props.value} />
        )}
      </div>
      {props.sub && <div className="sub">{props.sub}</div>}
    </div>
  );
}

/**
 * Break-even win rate against the realized one: the single number that says
 * whether the strategy sits on the right side of its own arithmetic. Below
 * break-even it loses money however good the entries look.
 *
 * When the outcome distribution is not binary the figure would be
 * meaningless, so the reason is shown in its place instead of a number
 * nobody should act on.
 */
function BreakevenPanel({
  breakeven,
  realized,
}: {
  breakeven: Breakeven;
  realized: number | null;
}) {
  if (!breakeven.valid) {
    const nonBinary = Object.entries(breakeven.non_binary_reasons ?? {});
    return (
      <Panel title="Break-even win rate">
        <Notice kind="info" title="Not applicable to this run">
          {breakeven.reason ?? "the outcome distribution is not binary"}
        </Notice>
        <dl className="facts" style={{ marginTop: 12 }}>
          <dt>Trades examined</dt>
          <dd>{int(breakeven.observations)}</dd>
          <dt>Realized win rate</dt>
          <dd>{realized === null ? "—" : pct(realized)}</dd>
          <dt>Non stop/target exits</dt>
          <dd>
            {int(breakeven.non_binary_trades)}
            {nonBinary.length > 0 &&
              ` (${nonBinary.map(([reason, count]) => `${reason} ${count}`).join(", ")})`}
          </dd>
        </dl>
      </Panel>
    );
  }

  const delta = breakeven.delta ?? null;
  const clears = delta !== null && delta > 0;
  return (
    <Panel
      title="Break-even win rate"
      aside={
        <Badge kind={clears ? "ok" : "bad"}>
          {clears ? "above break-even" : "below break-even"}
        </Badge>
      }
    >
      <div className="stack">
        <div className="pair">
          <div>
            <div className="label">Break-even</div>
            <div className="value">{pct(breakeven.breakeven_win_rate)}</div>
            <div className="sub">required to stop losing</div>
          </div>
          <div>
            <div className="label">Realized</div>
            <div className="value">
              <Signed value={delta} text={pct(breakeven.realized_win_rate)} />
            </div>
            <div className="sub">
              {delta === null ? "—" : `${signedPct(delta)} vs break-even`}
            </div>
          </div>
        </div>

        <dl className="facts">
          <dt>Trades examined</dt>
          <dd>{int(breakeven.observations)}</dd>
          <dt>Average win / loss</dt>
          <dd>
            {money(breakeven.avg_win)} / {money(breakeven.avg_loss)}
          </dd>
          <dt>Non stop/target exits</dt>
          <dd>{int(breakeven.non_binary_trades)}</dd>
        </dl>

        <p className="footnote">
          Stop and target are anchored to the spread-inclusive entry price, so the spread does
          not enter the cash amounts: it pushes the achievable win rate down instead of raising
          this threshold.
        </p>
      </div>
    </Panel>
  );
}

function EquitySection({
  equity,
  initial,
  runId,
}: {
  equity: Equity | null;
  initial: number;
  runId: string;
}) {
  // Drawn from the left the first time this run's curve appears, and never
  // again. A chart that re-animates on every resize or hover is one nobody
  // can read while they are working.
  const draw = useFirstDraw(runId);
  const data: ChartPoint[] = useMemo(
    () =>
      (equity?.points ?? []).map((point) => ({
        ts: new Date(point.t).getTime(),
        equity: point.equity,
        drawdown: point.drawdown,
      })),
    [equity],
  );

  if (!equity) {
    return (
      <Panel title="Equity curve">
        <Loading label="Loading the curve…" height={260} />
      </Panel>
    );
  }
  if (data.length === 0) {
    return (
      <Panel title="Equity curve">
        <Empty title="No equity point for this run" />
      </Panel>
    );
  }

  const domain: [number, number] = [data[0]!.ts, data[data.length - 1]!.ts];

  return (
    <Panel
      title="Equity and drawdown"
      aside={
        <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          {int(equity.total_points)} points reduced to {int(equity.returned_points)} · drawdown
          computed on the full series
        </span>
      }
      tight
    >
      <div className="legend">
        <span>
          <span className="swatch" style={{ background: EQUITY_COLOR }} />
          Mark-to-market equity (account currency) — Y axis does not start at zero
        </span>
        <span>
          <span className="swatch" style={{ background: DRAWDOWN_COLOR }} />
          Drawdown from the previous peak (%)
        </span>
      </div>

      <div className="chart-frame" style={{ height: 260, padding: "8px 8px 0" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid stroke="var(--chart-grid)" />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={domain}
              tickFormatter={(value: number) => utcDate(new Date(value).toISOString())}
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
              y={initial}
              stroke="var(--chart-axis)"
              strokeDasharray="4 3"
              label={{ value: "start", position: "insideLeft", fontSize: 11, fill: "var(--chart-axis)" }}
            />
            <Tooltip content={<ChartTooltip />} isAnimationActive={false} />
            <Line
              type="monotone"
              dataKey="equity"
              stroke={EQUITY_COLOR}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={draw}
              animationDuration={DRAW_MS}
              animationEasing="ease-out"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-frame" style={{ height: 130, padding: "0 8px 8px" }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 16, bottom: 4, left: 8 }} syncId="run">
            <CartesianGrid stroke="var(--chart-grid)" />
            <XAxis
              dataKey="ts"
              type="number"
              scale="time"
              domain={domain}
              tickFormatter={(value: number) => utcDate(new Date(value).toISOString())}
              tick={{ fontSize: 11 }}
              minTickGap={48}
            />
            <YAxis
              tick={{ fontSize: 11 }}
              width={64}
              domain={["dataMin", 0]}
              tickFormatter={(value: number) => pct(value, 0)}
            />
            <Tooltip content={<ChartTooltip />} isAnimationActive={false} />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke={DRAWDOWN_COLOR}
              fill={DRAWDOWN_COLOR}
              fillOpacity={0.14}
              strokeWidth={1.2}
              isAnimationActive={draw}
              animationDuration={DRAW_MS}
              animationEasing="ease-out"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { dataKey?: string | number; value?: number }[];
  label?: number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip">
      <div className="t">{utcDateTime(new Date(label ?? 0).toISOString(), true)} UTC</div>
      <dl>
        {payload.map((entry) => (
          <ChartTooltipRow
            key={String(entry.dataKey)}
            name={String(entry.dataKey)}
            value={entry.value}
          />
        ))}
      </dl>
    </div>
  );
}

function ChartTooltipRow({ name, value }: { name: string; value?: number }) {
  const isPct = name === "drawdown";
  return (
    <>
      <dt>{isPct ? "drawdown" : "equity"}</dt>
      <dd>{isPct ? pct(value, 3) : money(value, 4)}</dd>
    </>
  );
}

const METRIC_ROWS: { key: keyof Metrics; label: string; kind: "pct" | "money" | "num" | "int" }[] =
  [
    { key: "final_equity", label: "Final equity", kind: "money" },
    { key: "total_return", label: "Total return", kind: "pct" },
    { key: "annual_return", label: "Annual return", kind: "pct" },
    { key: "sharpe", label: "Sharpe", kind: "num" },
    { key: "sortino", label: "Sortino", kind: "num" },
    { key: "max_drawdown_pct", label: "Max drawdown %", kind: "pct" },
    { key: "max_drawdown_money", label: "Max drawdown currency", kind: "money" },
    { key: "calmar", label: "Calmar", kind: "num" },
    { key: "trades", label: "Trades", kind: "int" },
    { key: "win_rate", label: "Win rate", kind: "pct" },
    { key: "profit_factor", label: "Profit factor", kind: "num" },
    { key: "expectancy", label: "Expectancy / trade", kind: "money" },
    { key: "exposure", label: "Exposure", kind: "pct" },
    { key: "max_losing_streak", label: "Max losing streak", kind: "int" },
    { key: "t_stat", label: "t-stat", kind: "num" },
    { key: "p_value", label: "p-value", kind: "num" },
  ];

function cell(metrics: Metrics | null, key: keyof Metrics, kind: string) {
  if (!metrics) return "—";
  const value = metrics[key];
  if (typeof value !== "number") return "—";
  if (kind === "pct") return pct(value);
  if (kind === "money") return money(value);
  if (kind === "int") return int(value);
  return num(value, 3);
}

function MetricsTable({
  strategy,
  benchmark,
}: {
  strategy: Metrics;
  benchmark: Metrics | null;
}) {
  // `costs` has an API-side default, so it is optional in the generated types
  const costs = strategy.costs ?? {};
  const benchCosts = benchmark?.costs ?? {};
  return (
    <Panel title="Metrics" tight>
      {benchmark?.drawdown_unsustainable && (
        <div style={{ padding: 16, paddingBottom: 0 }}>
          <Notice kind="warn" title="Buy &amp; hold is not a sustainable comparison">
            With the same sizing the buy &amp; hold equity goes below zero (drawdown{" "}
            {pct(benchmark.max_drawdown_pct)}): a margin call would have arrived well before
            the end. Its numbers are arithmetic, not an achievable result.
          </Notice>
        </div>
      )}
      {strategy.drawdown_unsustainable && (
        <div style={{ padding: 16, paddingBottom: 0 }}>
          <Notice kind="error" title="Drawdown beyond 100%">
            The strategy equity went below zero: the run would not have survived.
          </Notice>
        </div>
      )}
      <div className="table-scroll">
        <table>
          <caption>
            Strategy against buy &amp; hold over the same period and with the same costs.
          </caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col" className="num">
                {strategy.label}
              </th>
              <th scope="col" className="num">
                buy &amp; hold
              </th>
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map((row) => (
              <tr key={String(row.key)}>
                <th scope="row" style={{ fontWeight: 400 }}>
                  {row.label}
                </th>
                <td className="num">
                  {row.kind === "pct" || row.kind === "money" ? (
                    <Signed
                      value={strategy[row.key] as number}
                      text={cell(strategy, row.key, row.kind)}
                    />
                  ) : (
                    cell(strategy, row.key, row.kind)
                  )}
                </td>
                <td className="num" style={{ color: "var(--ink-soft)" }}>
                  {cell(benchmark, row.key, row.kind)}
                </td>
              </tr>
            ))}
            <tr>
              <th scope="row" style={{ fontWeight: 400 }}>
                Costs (spread / commission / swap)
              </th>
              <td className="num">
                {money(costs.spread)} / {money(costs.commission)} / {money(costs.swap)}
              </td>
              <td className="num" style={{ color: "var(--ink-soft)" }}>
                {benchmark
                  ? `${money(benchCosts.spread)} / ${money(
                      benchCosts.commission,
                    )} / ${money(benchCosts.swap)}`
                  : "—"}
              </td>
            </tr>
            <tr>
              <th scope="row" style={{ fontWeight: 400 }}>
                Gross → net PnL
              </th>
              <td className="num">
                <Signed value={costs.gross} text={signedMoney(costs.gross)} />
                {" → "}
                <Signed value={costs.net} text={signedMoney(costs.net)} />
              </td>
              <td className="num" style={{ color: "var(--ink-soft)" }}>
                {benchmark
                  ? `${signedMoney(benchCosts.gross)} → ${signedMoney(benchCosts.net)}`
                  : "—"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/**
 * The rejected signals are not listed here: the Risk gates panel below tells
 * that story with labels and shares instead of raw codes, and saying it twice
 * in two different vocabularies helps nobody.
 */
function ExecutionPanel({ run }: { run: RunDetail }) {
  const execution = run.execution;
  if (!execution) return null;
  const exits = Object.entries(execution.exit_reasons ?? {});
  return (
    <Panel title="Execution">
      <div className="stack">
        <dl className="facts">
          <dt>Raw signals</dt>
          <dd>
            {int(execution.signals_long)} long · {int(execution.signals_short)} short
          </dd>
          <dt>Executed trades</dt>
          <dd>{int(execution.trades)}</dd>
          <dt>Ambiguous trades</dt>
          <dd>
            {int(execution.ambiguous_trades)}{" "}
            {execution.ambiguous_trades > 0 && (
              <Badge kind="warn">SL and TP in the same bar</Badge>
            )}
          </dd>
          <dt>Trades across a session gap</dt>
          <dd>{int(execution.gap_crossing_trades)}</dd>
        </dl>

        <div>
          <h3>Exits by reason</h3>
          {exits.length === 0 ? (
            <div style={{ color: "var(--ink-soft)" }}>none</div>
          ) : (
            <dl className="facts">
              {exits.map(([reason, count]) => (
                <ExitRow key={reason} reason={reason} count={count} />
              ))}
            </dl>
          )}
        </div>

      </div>
    </Panel>
  );
}

function ExitRow({ reason, count }: { reason: string; count: number }) {
  return (
    <>
      <dt>{reason}</dt>
      <dd>{int(count)}</dd>
    </>
  );
}

function TradesPanel({ runId }: { runId: string }) {
  const [page, setPage] = useState<Trades | null>(null);
  const [offset, setOffset] = useState(0);
  const [ambiguousOnly, setAmbiguousOnly] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    setError(null);
    api.trades(runId, offset, PAGE_SIZE, ambiguousOnly).then(setPage).catch(setError);
  }, [runId, offset, ambiguousOnly]);

  if (error !== null) return <ErrorNotice error={error} />;

  return (
    <Panel
      title="Trades"
      aside={
        <label style={{ fontSize: 12, display: "flex", gap: 6, alignItems: "center" }}>
          <input
            type="checkbox"
            checked={ambiguousOnly}
            style={{ width: "auto" }}
            onChange={(event) => {
              setAmbiguousOnly(event.target.checked);
              setOffset(0);
            }}
          />
          ambiguous only
          {page ? ` (${int(page.ambiguous_total)})` : ""}
        </label>
      }
      tight
    >
      {!page ? (
        <div style={{ padding: 16 }}>
          <Loading label="Loading the trades…" height={160} />
        </div>
      ) : page.items.length === 0 ? (
        <Empty title={ambiguousOnly ? "No ambiguous trade" : "No trade"}>
          <p>
            {ambiguousOnly
              ? "No bar touched stop and target together: the run has no uncertain outcome."
              : "The risk gates or the signals produced no operation."}
          </p>
        </Empty>
      ) : (
        <>
          <div className="table-scroll" style={{ maxHeight: 460, overflowY: "auto" }}>
            <table>
              <caption>
                Highlighted rows are ambiguous trades: stop and target touched in the same bar,
                counted as a stop.
              </caption>
              <thead>
                <tr>
                  <th scope="col" className="num">
                    #
                  </th>
                  <th scope="col">Dir</th>
                  <th scope="col">Entry (UTC)</th>
                  <th scope="col" className="num">
                    Price
                  </th>
                  <th scope="col">Exit (UTC)</th>
                  <th scope="col" className="num">
                    Price
                  </th>
                  <th scope="col">Reason</th>
                  <th scope="col" className="num">
                    Lots
                  </th>
                  <th scope="col" className="num">
                    Spread
                  </th>
                  <th scope="col" className="num">
                    Net
                  </th>
                  <th scope="col" className="num">
                    R
                  </th>
                  <th scope="col">Notes</th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((trade) => (
                  <tr key={trade.index} className={trade.ambiguous ? "flagged" : undefined}>
                    <td className="num">{trade.index + 1}</td>
                    <td>{trade.direction > 0 ? "LONG" : "SHORT"}</td>
                    <td>{utcDateTime(trade.entry_time)}</td>
                    <td className="num">{num(trade.entry_price, 2)}</td>
                    <td>{utcDateTime(trade.exit_time)}</td>
                    <td className="num">{num(trade.exit_price, 2)}</td>
                    <td>{trade.exit_reason}</td>
                    <td className="num">{num(trade.lots, 2)}</td>
                    <td className="num">{num(trade.spread_cost, 3)}</td>
                    <td className="num">
                      <strong>
                        <Signed value={trade.net_pnl} text={signedMoney(trade.net_pnl, 3)} />
                      </strong>
                    </td>
                    <td className="num">{num(trade.r_multiple, 2)}</td>
                    <td>
                      {trade.ambiguous && <Badge kind="warn">SL+TP</Badge>}{" "}
                      {trade.crossed_gap && <Badge kind="mute">gap</Badge>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <button
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
            >
              ← previous
            </button>
            <span>
              {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {int(page.total)}
            </span>
            <button
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= page.total}
            >
              next →
            </button>
          </div>
        </>
      )}
    </Panel>
  );
}
