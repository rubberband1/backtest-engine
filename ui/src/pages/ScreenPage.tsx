import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type RunConfigIn,
  type ScreenCell,
  type ScreenJob,
  type ScreenReport,
  type Strategy,
  type SymbolList,
} from "../api/client";
import { Badge, Empty, ErrorNotice, Field, Loading, Notice, Panel, Signed } from "../components/ui";
import { int, num, pct, signedMoney } from "../format";
import { navigate } from "../router";

const TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"];
const POLL_MS = 1500;

type SortKey =
  | "strategy_id"
  | "symbol"
  | "timeframe"
  | "gate_signals"
  | "trades"
  | "net_pnl"
  | "sharpe_per_trade"
  | "mean_r"
  | "win_rate"
  | "ambiguous_share"
  | "permutation_p_value";

const COLUMNS: { key: SortKey; label: string; num: boolean }[] = [
  { key: "strategy_id", label: "Strategy", num: false },
  { key: "symbol", label: "Instrument", num: false },
  { key: "timeframe", label: "TF", num: false },
  { key: "gate_signals", label: "Signals", num: true },
  { key: "trades", label: "Trades", num: true },
  { key: "net_pnl", label: "Net PnL", num: true },
  { key: "sharpe_per_trade", label: "Sharpe/trade", num: true },
  { key: "mean_r", label: "Mean R", num: true },
  { key: "win_rate", label: "Win rate", num: true },
  { key: "ambiguous_share", label: "Ambiguous", num: true },
  { key: "permutation_p_value", label: "Perm. p", num: true },
];

/**
 * A campaign of hundreds of backtests is also a machine for producing false
 * positives: at the 5% level, three hundred attempts return about fifteen
 * "significant" results with no edge anywhere in the data. So the trial
 * panel is not a footnote at the bottom of this page - it sits above the
 * table, always visible, and every row is read against it.
 */
export function ScreenPage({ jobId }: { jobId: string | null }) {
  const [symbols, setSymbols] = useState<SymbolList | null>(null);
  const [strategies, setStrategies] = useState<Strategy[] | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);

  const [pickedStrategies, setPickedStrategies] = useState<string[]>([]);
  const [pickedSymbols, setPickedSymbols] = useState<string[]>([]);
  const [pickedTimeframes, setPickedTimeframes] = useState<string[]>(["M5", "M15", "H1"]);
  const [equity, setEquity] = useState(100);
  const [minTrades, setMinTrades] = useState(30);
  const [iterations, setIterations] = useState(200);

  const [job, setJob] = useState<ScreenJob | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [sort, setSort] = useState<{ key: SortKey; asc: boolean }>({
    key: "sharpe_per_trade",
    asc: false,
  });
  const [onlyBacktested, setOnlyBacktested] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    Promise.all([api.symbols(), api.strategies()])
      .then(([symbolList, strategyList]) => {
        setSymbols(symbolList);
        setStrategies(strategyList);
        setPickedStrategies(strategyList.map((item) => item.id));
        setPickedSymbols((symbolList.cached_symbols ?? []).slice(0, 10));
      })
      .catch(setLoadError);
  }, []);

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, []);

  // a campaign in the URL is reattached on load: the job lives in the backend
  // and a reload should find it again, not lose twenty minutes of work
  useEffect(() => {
    if (jobId) poll(jobId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  function poll(id: string) {
    api
      .screenJob(id)
      .then((next) => {
        setJob(next);
        if (next.status === "running") {
          timer.current = window.setTimeout(() => poll(id), POLL_MS);
        }
      })
      .catch(setError);
  }

  async function start() {
    setError(null);
    setJob(null);
    try {
      const config: RunConfigIn = {
        symbol: pickedSymbols[0] ?? "",
        timeframe: pickedTimeframes[0] ?? "H1",
        start: null,
        end: null,
        initial_equity: equity,
        spread_mode: "per_bar",
        spread_value: null,
        commission_per_lot_per_side: 0,
        swap_mode: "points",
        session_threshold: 0.5,
      };
      const started = await api.screen({
        strategy_ids: pickedStrategies,
        symbols: pickedSymbols,
        timeframes: pickedTimeframes,
        config,
        min_trades: minTrades,
        permutation_iterations: iterations,
      });
      setJob(started);
      navigate(`/screen?job=${started.job_id}`);
      poll(started.job_id);
    } catch (problem) {
      setError(problem);
    }
  }

  const report = job?.report ?? null;
  const cells = useMemo(() => {
    const rows = [...(report?.cells ?? [])];
    const filtered = onlyBacktested ? rows.filter((cell) => cell.trades !== null) : rows;
    filtered.sort((left, right) => {
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
    return filtered;
  }, [report, sort, onlyBacktested]);

  if (loadError !== null) return <ErrorNotice error={loadError} />;
  if (!symbols || !strategies) return <Loading label="Loading strategies and instruments…" />;

  const cached = symbols.cached_symbols ?? [];
  const total = pickedStrategies.length * pickedSymbols.length * pickedTimeframes.length;
  const running = job?.status === "running";

  return (
    <div className="stack">
      <Panel
        title="Screening campaign"
        aside={
          <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
            gate zero → backtest → permutation, in that order
          </span>
        }
      >
        <div className="stack">
          <div className="form-grid">
            <Field label="Initial equity" htmlFor="s-equity" hint="per cell, account currency">
              <input
                id="s-equity"
                type="number"
                min={1}
                value={equity}
                onChange={(event) => setEquity(Number(event.target.value))}
              />
            </Field>
            <Field
              label="Minimum trades"
              htmlFor="s-min"
              hint="below this a backtest has no power, whatever its curve"
            >
              <input
                id="s-min"
                type="number"
                min={1}
                value={minTrades}
                onChange={(event) => setMinTrades(Number(event.target.value))}
              />
            </Field>
            <Field
              label="Permutation iterations"
              htmlFor="s-iter"
              hint="reduced on purpose: a campaign is not a confirmation"
            >
              <input
                id="s-iter"
                type="number"
                min={10}
                max={5000}
                value={iterations}
                onChange={(event) => setIterations(Number(event.target.value))}
              />
            </Field>
            {/* laid out in a row, not in a scroller: a selected timeframe
                hidden below the fold is a setting the user cannot see */}
            <Field label="Timeframes" hint="the spec's own timeframe is overridden per cell">
              <div className="chips">
                {TIMEFRAMES.map((name) => (
                  <label key={name}>
                    <input
                      type="checkbox"
                      checked={pickedTimeframes.includes(name)}
                      onChange={() =>
                        setPickedTimeframes((current) =>
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
            </Field>
          </div>

          <div className="row">
            <div className="grow">
              <h3>Strategies ({pickedStrategies.length})</h3>
              <div className="checklist" style={{ maxHeight: 200 }}>
                {strategies.map((item) => (
                  <label key={item.id}>
                    <input
                      type="checkbox"
                      checked={pickedStrategies.includes(item.id)}
                      onChange={() =>
                        setPickedStrategies((current) =>
                          current.includes(item.id)
                            ? current.filter((entry) => entry !== item.id)
                            : [...current, item.id],
                        )
                      }
                    />
                    <span className="mono">{item.id}</span>
                  </label>
                ))}
              </div>
            </div>
            <div className="grow">
              <h3>Instruments with cached data ({pickedSymbols.length})</h3>
              {cached.length === 0 ? (
                <Empty title="No instrument has cached data">
                  <p>
                    Download history with{" "}
                    <code>python -m examples.download_year &lt;SYMBOL&gt;</code> first.
                  </p>
                </Empty>
              ) : (
                <div className="checklist" style={{ maxHeight: 200 }}>
                  {cached.map((name) => (
                    <label key={name}>
                      <input
                        type="checkbox"
                        checked={pickedSymbols.includes(name)}
                        onChange={() =>
                          setPickedSymbols((current) =>
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
          </div>

          <div className="actions">
            <button className="primary" onClick={start} disabled={running || total === 0}>
              {running ? "Campaign running…" : `Screen ${total} cells`}
            </button>
            <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
              {total} attempts is {total} chances to be lucky: the threshold below is
              corrected for exactly that number.
            </span>
          </div>
        </div>
      </Panel>

      {error !== null && <ErrorNotice error={error} />}
      {job?.status === "error" && (
        <Notice kind="error" title="The campaign failed">
          {job.error}
        </Notice>
      )}

      {running && (
        <Panel title="Campaign running">
          <div className="stack">
            <progress value={job.completed_cells} max={job.total_cells || 1} style={{ width: "100%" }} />
            <div style={{ color: "var(--ink-soft)", fontSize: 13 }}>
              {int(job.completed_cells)} of {int(job.total_cells)} cells ·{" "}
              {job.current ?? "starting"}
            </div>
            <Loading label="Gate zero on every cell, backtests on the survivors, permutations last." height={80} />
          </div>
        </Panel>
      )}

      {report && (
        <>
          <TrialPanel report={report} />
          <Panel
            title="Strategy × instrument × timeframe"
            aside={
              <label style={{ fontSize: 12, color: "var(--ink-soft)" }}>
                <input
                  type="checkbox"
                  checked={onlyBacktested}
                  onChange={() => setOnlyBacktested((value) => !value)}
                />{" "}
                only cells that reached a backtest
              </label>
            }
            tight
          >
            <div className="table-scroll">
              <table>
                <caption>
                  Every row is one attempt. A row that looks good is marked{" "}
                  <em>not significant</em> unless its Sharpe clears the corrected
                  threshold above — the number to read is that one, not the equity.
                </caption>
                <thead>
                  <tr>
                    {COLUMNS.map((column) => (
                      <th key={column.key} scope="col" className={column.num ? "num" : undefined}>
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
                    <th scope="col">Stage</th>
                    <th scope="col">Run</th>
                  </tr>
                </thead>
                <tbody>
                  {cells.map((cell) => (
                    <CellRow
                      key={`${cell.strategy_id}-${cell.symbol}-${cell.timeframe}`}
                      cell={cell}
                      required={report.panel.required_sharpe_per_trade}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}

      {!job && (
        <Empty title="No campaign yet">
          <p>
            Pick strategies, instruments and timeframes, then screen them. Cells stopped at
            gate zero still count as attempts: that is the whole point of the panel.
          </p>
        </Empty>
      )}
    </div>
  );
}

function CellRow({ cell, required }: { cell: ScreenCell; required: number | null | undefined }) {
  const backtested = cell.trades !== null && cell.trades !== undefined;
  const clears =
    backtested &&
    required !== null &&
    required !== undefined &&
    cell.sharpe_per_trade !== null &&
    cell.sharpe_per_trade !== undefined &&
    cell.sharpe_per_trade >= required;
  const promising =
    backtested && (cell.net_pnl ?? 0) > 0 && (cell.sharpe_per_trade ?? 0) > 0;

  return (
    <tr className={cell.status === "error" ? "flagged" : undefined}>
      <th scope="row" className="mono" style={{ fontWeight: 400 }}>
        {cell.strategy_id}
      </th>
      <td className="mono">{cell.symbol}</td>
      <td className="mono">{cell.timeframe}</td>
      <td className="num">{int(cell.gate_signals)}</td>
      <td className="num">{backtested ? int(cell.trades) : "—"}</td>
      <td className="num">
        {backtested ? <Signed value={cell.net_pnl} text={signedMoney(cell.net_pnl)} /> : "—"}
      </td>
      <td className="num">
        {backtested ? (
          <Signed value={cell.sharpe_per_trade} text={signedMoney(cell.sharpe_per_trade, 4)} />
        ) : (
          "—"
        )}
      </td>
      <td className="num">
        {backtested ? <Signed value={cell.mean_r} text={signedMoney(cell.mean_r, 3)} /> : "—"}
      </td>
      <td className="num">{backtested ? pct(cell.win_rate) : "—"}</td>
      <td className="num">
        {backtested ? (
          <span className={(cell.ambiguous_share ?? 0) > 0.05 ? "neg" : undefined}>
            {pct(cell.ambiguous_share, 1)}
          </span>
        ) : (
          "—"
        )}
      </td>
      <td className="num">{num(cell.permutation_p_value, 4)}</td>
      <td>
        {cell.status === "error" ? (
          <Badge kind="bad">error</Badge>
        ) : cell.stage_reached === "gate" ? (
          <Badge kind="mute">stopped at gate zero</Badge>
        ) : promising && !clears ? (
          <Badge kind="warn">not significant</Badge>
        ) : clears ? (
          <Badge kind="ok">clears the threshold</Badge>
        ) : (
          <Badge kind="mute">{cell.stage_reached}</Badge>
        )}
      </td>
      <td className="mono">
        {cell.run_id ? <a href={`#/result/${cell.run_id}`}>{cell.run_id.slice(0, 8)}</a> : "—"}
      </td>
    </tr>
  );
}

/** Always above the table: the count of attempts and what it costs. */
function TrialPanel({ report }: { report: ScreenReport }) {
  const panel = report.panel;
  return (
    <Panel
      title="Multiple testing over the whole campaign"
      aside={
        panel.best_clears_required === true ? (
          <Badge kind="ok">the best result clears the corrected threshold</Badge>
        ) : (
          <Badge kind="warn">nothing clears the corrected threshold</Badge>
        )
      }
    >
      <div className="stack">
        <Notice kind={panel.best_clears_required === true ? "ok" : "warn"} title="Verdict">
          {panel.verdict}
        </Notice>

        <div className="kpis">
          <div className="kpi">
            <div className="label">Attempts</div>
            <div className="value">{int(panel.attempts)}</div>
            <div className="sub">
              {int(panel.cells_backtested)} backtested · {int(panel.cells_permuted)} permuted
            </div>
          </div>
          <div className="kpi">
            <div className="label">Free Sharpe at N={int(panel.attempts)}</div>
            <div className="value small">{signedMoney(panel.expected_max_sharpe, 4)}</div>
            <div className="sub">what the best of N trials reaches by luck alone</div>
          </div>
          <div className="kpi">
            <div className="label">Required Sharpe/trade</div>
            <div className="value small">{signedMoney(panel.required_sharpe_per_trade, 4)}</div>
            <div className="sub">to be credible at {pct(panel.confidence, 0)} confidence</div>
          </div>
          <div className="kpi">
            <div className="label">Best observed</div>
            <div className="value small">
              <Signed
                value={panel.best_sharpe_per_trade}
                text={signedMoney(panel.best_sharpe_per_trade, 4)}
              />
            </div>
            <div className="sub">{panel.best_strategy ?? "—"}</div>
          </div>
          <div className="kpi">
            <div className="label">Bonferroni α</div>
            <div className="value small">{num(panel.bonferroni_threshold, 5)}</div>
            <div className="sub">
              {int(panel.survivors_after_correction)} cells survive it with a positive PnL
            </div>
          </div>
        </div>

        {(report.warnings ?? []).map((warning) => (
          <Notice key={warning} kind="warn" title="Careful">
            {warning}
          </Notice>
        ))}

        <details>
          <summary>What this correction assumes</summary>
          <ul className="footnote">
            {(panel.assumptions ?? []).map((assumption) => (
              <li key={assumption}>{assumption}</li>
            ))}
          </ul>
        </details>
      </div>
    </Panel>
  );
}
