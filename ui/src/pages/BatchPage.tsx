import { useEffect, useMemo, useState } from "react";
import {
  api,
  type BatchResponse,
  type BatchRunOut,
  type RunConfigIn,
  type Strategy,
  type SymbolList,
} from "../api/client";
import { Badge, Empty, ErrorNotice, Field, Loading, Notice, Panel, Signed } from "../components/ui";
import { int, num, pct, signedMoney } from "../format";

type SortKey = keyof Pick<
  BatchRunOut,
  | "symbol"
  | "trades"
  | "net_pnl"
  | "expectancy"
  | "mean_r"
  | "sharpe"
  | "profit_factor"
  | "win_rate"
  | "max_drawdown_pct"
>;

const COLUMNS: { key: SortKey; label: string; kind: "text" | "int" | "money" | "num" | "pct" }[] = [
  { key: "symbol", label: "Instrument", kind: "text" },
  { key: "trades", label: "Trades", kind: "int" },
  { key: "net_pnl", label: "Net PnL", kind: "money" },
  { key: "expectancy", label: "Expectancy", kind: "num" },
  { key: "mean_r", label: "Mean R", kind: "num" },
  { key: "sharpe", label: "Sharpe", kind: "num" },
  { key: "profit_factor", label: "Profit factor", kind: "num" },
  { key: "win_rate", label: "Win rate", kind: "pct" },
  { key: "max_drawdown_pct", label: "Max DD", kind: "pct" },
];

/**
 * One instrument is one experiment. This page exists because a result on a
 * single symbol cannot tell an edge from that symbol's history, and the only
 * cure is to ask the same question of many instruments and look at the
 * spread of the answers.
 */
export function BatchPage() {
  const [symbols, setSymbols] = useState<SymbolList | null>(null);
  const [strategies, setStrategies] = useState<Strategy[] | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);

  const [strategyId, setStrategyId] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [timeframe, setTimeframe] = useState("M1");
  const [equity, setEquity] = useState(100);
  const [metric, setMetric] = useState<"mean_r" | "expectancy" | "net_pnl" | "sharpe">("mean_r");

  const [report, setReport] = useState<BatchResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({ key: "mean_r", asc: false });

  useEffect(() => {
    Promise.all([api.symbols(), api.strategies()])
      .then(([symbolList, strategyList]) => {
        setSymbols(symbolList);
        setStrategies(strategyList);
        const first = strategyList[0];
        if (first) {
          setStrategyId(first.id);
          setTimeframe(first.timeframe);
        }
        setPicked((symbolList.cached_symbols ?? []).slice(0, 10));
      })
      .catch(setLoadError);
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    setReport(null);
    try {
      const config: RunConfigIn = {
        symbol: picked[0] ?? "",
        timeframe,
        start: null,
        end: null,
        initial_equity: equity,
        spread_mode: "per_bar",
        spread_value: null,
        commission_per_lot_per_side: 0,
        swap_mode: "points",
        session_threshold: 0.5,
      };
      setReport(
        await api.batch({ strategy_id: strategyId, symbols: picked, config }),
      );
    } catch (problem) {
      setError(problem);
    } finally {
      setBusy(false);
    }
  }

  const rows = useMemo(() => {
    if (!report) return [];
    const copy = [...(report.cells ?? [])];
    copy.sort((left, right) => {
      const a = left[sort.key];
      const b = right[sort.key];
      if (typeof a === "string" || typeof b === "string") {
        return sort.asc
          ? String(a).localeCompare(String(b))
          : String(b).localeCompare(String(a));
      }
      const x = typeof a === "number" ? a : Number.NEGATIVE_INFINITY;
      const y = typeof b === "number" ? b : Number.NEGATIVE_INFINITY;
      return sort.asc ? x - y : y - x;
    });
    return copy;
  }, [report, sort]);

  if (loadError !== null) return <ErrorNotice error={loadError} />;
  if (!symbols || !strategies) return <Loading label="Loading symbols and strategies…" />;

  const cached = symbols.cached_symbols ?? [];

  return (
    <div className="stack">
      <Panel
        title="Batch across instruments"
        aside={
          <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            every cell is a normal run and lands in runs/
          </span>
        }
      >
        <div className="stack">
          <div className="form-grid">
            <Field label="Strategy" htmlFor="b-strategy">
              <select
                id="b-strategy"
                value={strategyId}
                onChange={(event) => setStrategyId(event.target.value)}
              >
                {strategies.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} ({item.id})
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Timeframe" htmlFor="b-tf">
              <input
                id="b-tf"
                type="text"
                value={timeframe}
                onChange={(event) => setTimeframe(event.target.value)}
              />
            </Field>
            <Field label="Initial equity" htmlFor="b-equity" hint="per cell, account currency">
              <input
                id="b-equity"
                type="number"
                min={1}
                value={equity}
                onChange={(event) => setEquity(Number(event.target.value))}
              />
            </Field>
            <Field
              label="Consistency metric"
              htmlFor="b-metric"
              hint="mean R is comparable across instruments"
            >
              <select
                id="b-metric"
                value={metric}
                onChange={(event) => setMetric(event.target.value as typeof metric)}
              >
                <option value="mean_r">mean R multiple</option>
                <option value="expectancy">expectancy per trade</option>
                <option value="net_pnl">net PnL</option>
                <option value="sharpe">Sharpe</option>
              </select>
            </Field>
          </div>

          <div>
            <h3>Instruments with cached data ({cached.length})</h3>
            {cached.length === 0 ? (
              <Empty title="No instrument has cached data">
                <p>
                  Download history with{" "}
                  <code>python -m examples.download_year &lt;SYMBOL&gt;</code> before running a
                  batch.
                </p>
              </Empty>
            ) : (
              <div className="checklist" style={{ maxHeight: 220 }}>
                {cached.map((name) => (
                  <label key={name}>
                    <input
                      type="checkbox"
                      checked={picked.includes(name)}
                      onChange={() =>
                        setPicked((current) =>
                          current.includes(name)
                            ? current.filter((item) => item !== name)
                            : [...current, name],
                        )
                      }
                    />
                    <span className="mono">{name}</span>
                  </label>
                ))}
              </div>
            )}
          </div>

          <div className="actions">
            <button
              className="primary"
              onClick={run}
              disabled={busy || picked.length === 0 || !strategyId}
            >
              {busy ? `Running ${picked.length} cells…` : `Run batch on ${picked.length}`}
            </button>
            <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
              Below three instruments the cross-sectional check has no power.
            </span>
          </div>
        </div>
      </Panel>

      {error !== null && <ErrorNotice error={error} />}
      {busy && (
        <Panel title="Batch running">
          <Loading
            label={`Running ${picked.length} backtests in parallel. Each one is a full run.`}
            height={220}
          />
        </Panel>
      )}

      {report && (
        <>
          <ConsistencyPanel report={report} />
          <Panel
            title="Instrument by metric"
            aside={
              <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
                {int(report.completed)} completed · {int(report.failed)} failed ·{" "}
                {num(report.elapsed_seconds, 1)} s
              </span>
            }
            tight
          >
            <div className="table-scroll">
              <table>
                <caption>
                  Click a column to sort. The standard error next to mean R is what says
                  whether a per-instrument figure is a measurement or a rounding of noise.
                </caption>
                <thead>
                  <tr>
                    {COLUMNS.map((column) => (
                      <th
                        key={column.key}
                        scope="col"
                        className={column.kind === "text" ? undefined : "num"}
                      >
                        <button
                          className="sort"
                          onClick={() =>
                            setSort((current) => ({
                              key: column.key,
                              asc: current.key === column.key ? !current.asc : false,
                            }))
                          }
                          aria-label={`Sort by ${column.label}`}
                        >
                          {column.label}
                          {sort.key === column.key ? (sort.asc ? " ▲" : " ▼") : ""}
                        </button>
                      </th>
                    ))}
                    <th scope="col" className="num">
                      Mean R ± se
                    </th>
                    <th scope="col">Run</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((cell) => (
                    <tr
                      key={`${cell.symbol}-${cell.run_id ?? "error"}`}
                      className={cell.status !== "done" ? "flagged" : undefined}
                    >
                      <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                        {cell.symbol}
                      </th>
                      {cell.status !== "done" ? (
                        <td colSpan={COLUMNS.length} style={{ whiteSpace: "normal" }}>
                          <Badge kind="bad">failed</Badge> {cell.error}
                        </td>
                      ) : (
                        <>
                          <td className="num">{int(cell.trades)}</td>
                          <td className="num">
                            <Signed value={cell.net_pnl} text={signedMoney(cell.net_pnl)} />
                          </td>
                          <td className="num">
                            <Signed
                              value={cell.expectancy}
                              text={signedMoney(cell.expectancy, 4)}
                            />
                          </td>
                          <td className="num">
                            <strong>
                              <Signed value={cell.mean_r} text={signedMoney(cell.mean_r, 3)} />
                            </strong>
                          </td>
                          <td className="num">{num(cell.sharpe, 2)}</td>
                          <td className="num">{num(cell.profit_factor, 3)}</td>
                          <td className="num">{pct(cell.win_rate)}</td>
                          <td className="num">{pct(cell.max_drawdown_pct)}</td>
                        </>
                      )}
                      <td className="num" style={{ color: "var(--ink-soft)" }}>
                        {cell.mean_r_stderr === null || cell.mean_r_stderr === undefined
                          ? "—"
                          : `± ${num(cell.mean_r_stderr, 3)}`}
                      </td>
                      <td className="mono">
                        {cell.run_id ? (
                          <a href={`#/result/${cell.run_id}`}>{cell.run_id.slice(0, 8)}</a>
                        ) : (
                          "—"
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}

function ConsistencyPanel({ report }: { report: BatchResponse }) {
  const measure = report.consistency;
  const agrees = measure.positive > measure.negative;
  return (
    <Panel
      title="Cross-sectional consistency"
      aside={
        measure.carried_by_one_symbol ? (
          <Badge kind="bad">carried by one instrument</Badge>
        ) : agrees ? (
          <Badge kind="ok">majority positive</Badge>
        ) : (
          <Badge kind="warn">majority negative</Badge>
        )
      }
    >
      <div className="stack">
        <Notice kind={measure.carried_by_one_symbol ? "warn" : "info"} title="What the instruments say">
          {measure.verdict}
        </Notice>
        {(report.warnings ?? []).map((warning) => (
          <Notice key={warning} kind="warn" title="Careful">
            {warning}
          </Notice>
        ))}

        <div className="pair">
          <div>
            <div className="label">Sign agreement</div>
            <div className="value">
              {int(measure.positive)} / {int(measure.observations)}
            </div>
            <div className="sub">instruments with a positive {measure.metric}</div>
          </div>
          <div>
            <div className="label">Mean {measure.metric}</div>
            <div className="value">
              <Signed value={measure.mean} text={signedMoney(measure.mean, 4)} />
            </div>
            <div className="sub">
              {measure.stderr === null || measure.stderr === undefined
                ? "standard error needs at least two instruments"
                : `± ${num(measure.stderr, 4)} across instruments`}
            </div>
          </div>
          <div>
            <div className="label">Sign test</div>
            <div className="value">{num(measure.sign_p_value, 3)}</div>
            <div className="sub">probability of a split this lopsided under a fair coin</div>
          </div>
        </div>

        <dl className="facts">
          <dt>Instruments with a result</dt>
          <dd>
            {int(measure.observations)} ({int(measure.positive)} positive,{" "}
            {int(measure.negative)} negative, {int(measure.zero_or_missing)} without a value)
          </dd>
          <dt>Spread across instruments</dt>
          <dd>
            min {num(measure.minimum, 4)} · median {num(measure.median, 4)} · max{" "}
            {num(measure.maximum, 4)}
            {measure.std !== null && measure.std !== undefined && ` · std ${num(measure.std, 4)}`}
          </dd>
          <dt>Cells</dt>
          <dd>
            {int(report.completed)} completed of {int((report.cells ?? []).length)}, grid size{" "}
            {int(report.grid_size)}
          </dd>
        </dl>
      </div>
    </Panel>
  );
}
