import { NumberField } from "../components/NumberField";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  type RunConfigIn,
  type ScreenCell,
  type ScreenJob,
  type ScreenReport,
  type Strategy,
  type SymbolList,
  type Tradability,
  type TradabilityCell,
} from "../api/client";
import { Progress } from "../components/Progress";
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
  | "gate_rejected_share"
  | "spread_measured_share"
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
  { key: "gate_rejected_share", label: "Rejected by gates", num: true },
  // how much of this cell's cost was measured. A campaign whose cells sit at
  // 0% is not wrong, but every number in it rests on a spread taken from a
  // period the cell does not cover, and that belongs on the row rather than
  // in a caveat under the table.
  { key: "spread_measured_share", label: "Spread measured", num: true },
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

  const [tradability, setTradability] = useState<Tradability | null>(null);
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
    // stage zero does not need a campaign to be useful: it answers "what is
    // worth trading here at all" on its own, so it loads with the page
    api.tradability().then(setTradability).catch(() => setTradability(null));
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
    // above M1 a per-bar spread is rebuilt from the M1 sample; the
    // median is the point of that distribution a fill is charged at
    per_bar_spread_quantile: 0.5,
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
              <NumberField
                id="s-equity"
                min={1}
                value={equity}
                onChange={(value) => setEquity(value ?? 0)}
              />
            </Field>
            <Field
              label="Minimum trades"
              htmlFor="s-min"
              hint="below this a backtest has no power, whatever its curve"
            >
              <NumberField
                id="s-min"
                min={1}
                value={minTrades}
                onChange={(value) => setMinTrades(value ?? 1)}
              />
            </Field>
            <Field
              label="Permutation iterations"
              htmlFor="s-iter"
              hint="reduced on purpose: a campaign is not a confirmation"
            >
              <NumberField
                id="s-iter"
                min={10}
                max={5000}
                value={iterations}
                onChange={(value) => setIterations(value ?? 10)}
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

      <TradabilityPanel table={tradability} />

      {error !== null && <ErrorNotice error={error} />}
      {job?.status === "error" && (
        <Notice kind="error" title="The campaign failed">
          {job.error}
        </Notice>
      )}

      {running && (
        <Panel title="Campaign running">
          <div className="stack">
            <Progress
              label="screening cell"
              detail={
                job.current
                  ? `${job.current} · gate zero first, a backtest on the survivors, permutations last`
                  : "gate zero on every cell, backtests on the survivors, permutations last"
              }
              completed={job.completed_cells}
              total={job.total_cells}
            />
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
      <td className="num">
        {cell.gate_rejected_share === null || cell.gate_rejected_share === undefined ? (
          "—"
        ) : (
          <span
            className={cell.gates_materially_altered ? "neg" : undefined}
            title={cell.relaxed_verdict ?? undefined}
          >
            {pct(cell.gate_rejected_share, 0)}
            {cell.gates_materially_altered ? " !" : ""}
          </span>
        )}
      </td>
      <td className="num">
        {cell.spread_measured_share === null || cell.spread_measured_share === undefined ? (
          "—"
        ) : (
          <span
            className={cell.spread_measured_share <= 0 ? "neg" : undefined}
            title={
              cell.spread_measured_share <= 0
                ? "every bar was charged a spread measured on a different period"
                : `${cell.spread_assumed_bars ?? 0} bars on an assumed spread`
            }
          >
            {pct(cell.spread_measured_share, 0)}
            {cell.spread_measured_share <= 0 ? " !" : ""}
          </span>
        )}
      </td>
      <td className="num">{num(cell.permutation_p_value, 4)}</td>
      <td>
        {cell.status === "error" ? (
          <Badge kind="bad">error</Badge>
        ) : cell.counts_as_attempt === false ? (
          <Badge kind="mute">refused before testing</Badge>
        ) : cell.stage_reached === "gate" ? (
          <Badge kind="mute">stopped at gate zero</Badge>
        ) : cell.gates_materially_altered ? (
          <Badge kind="warn">altered by its gates</Badge>
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
      className="reveal"
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

/**
 * Stage zero, and the first thing on this page.
 *
 * The question it answers comes before any strategy: on which of this
 * broker's instruments, at which timeframes, is algorithmic trading possible
 * at all? Where the spread is a large enough share of the typical stop, the
 * answer is no, and running a campaign there produces numbers about a bet
 * that cannot be won.
 *
 * The spread is measured on M1 and never on the bars being judged: above M1
 * the `spread` column of a bar is the minimum spread inside it, which on this
 * broker is zero on most FX hours and would make every cell look free.
 */
function TradabilityPanel({ table }: { table: Tradability | null }) {
  const [expanded, setExpanded] = useState(false);
  if (!table) {
    return (
      <Panel title="Tradability">
        <Loading label="Measuring spread against volatility on every cached pair." height={90} />
      </Panel>
    );
  }

  const cells = table.cells ?? [];
  const refused = cells.filter((cell) => !cell.tradable);
  const unjudged = cells.filter((cell) => !cell.judged);
  const shown = expanded ? cells : [...refused, ...unjudged];

  return (
    <Panel
      title="Tradability — what is worth testing at all"
      aside={
        refused.length > 0 ? (
          <Badge kind="warn">
            {int(refused.length)} of {int(cells.length)} cells refused
          </Badge>
        ) : (
          <Badge kind="ok">every cell is testable</Badge>
        )
      }
    >
      <div className="stack">
        <Notice kind="info" title="How a cell is refused">
          A pair is not testable when its median spread exceeds{" "}
          <strong>{pct(table.max_ratio, 0)}</strong> of one ATR({table.atr_period}) — that is{" "}
          {pct(table.max_ratio / table.stop_atr_mult, 1)} of the {num(table.stop_atr_mult, 0)}×ATR
          stop every strategy in this library places. The spread is measured on M1, where the
          field is a spread; on a coarser bar it is the <em>minimum</em> spread inside the bar,
          which is not a cost anyone pays. Refused cells are not counted as attempts in the
          correction below: nothing was tested on them.
        </Notice>

        <div className="kpis">
          <div className="kpi">
            <div className="label">Testable</div>
            <div className="value">{int(table.tradable)}</div>
            <div className="sub">of {int(cells.length)} instrument × timeframe pairs</div>
          </div>
          <div className="kpi">
            <div className="label">Refused</div>
            <div className="value">{int(table.excluded)}</div>
            <div className="sub">spread above the threshold: not an experiment</div>
          </div>
          <div className="kpi">
            <div className="label">Unjudged</div>
            <div className="value">{int(table.unjudged)}</div>
            <div className="sub">no measurement, so no verdict either way</div>
          </div>
        </div>

        {shown.length === 0 ? (
          <Empty title="Every measured cell is testable">
            <p>
              No pair hands the broker more than {pct(table.max_ratio, 0)} of one ATR. Open the
              full table to see how much each one does hand over.
            </p>
          </Empty>
        ) : (
          <div className="table-scroll">
            <table>
              <caption>
                {expanded
                  ? "Every cached instrument at every timeframe."
                  : "Only the cells that were refused or could not be judged. Open the full table for the rest."}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Instrument</th>
                  <th scope="col">TF</th>
                  <th scope="col" className="num">Spread (pt)</th>
                  <th scope="col" className="num">ATR (pt)</th>
                  <th scope="col" className="num">Spread / ATR</th>
                  <th scope="col" className="num">Share of stop</th>
                  <th scope="col">Verdict</th>
                  <th scope="col">Note</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((cell) => (
                  <TradabilityRow key={`${cell.symbol}-${cell.timeframe}`} cell={cell} />
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div>
          <button onClick={() => setExpanded((value) => !value)}>
            {expanded ? "Show only the refused cells" : `Show all ${int(cells.length)} cells`}
          </button>
        </div>

        {(table.warnings ?? []).map((warning) => (
          <Notice key={warning} kind="warn" title="Careful">
            {warning}
          </Notice>
        ))}
      </div>
    </Panel>
  );
}

/** The part of the reason the numeric columns do not already say. */
function note(cell: TradabilityCell): string {
  if (!cell.judged) return cell.reason;
  const extrapolated = cell.reason.includes("an assumption, not a measurement");
  if (!cell.tradable) {
    return extrapolated
      ? "above the threshold · spread measured after this series begins"
      : "above the threshold";
  }
  return extrapolated ? "spread measured after this series begins" : "";
}

function TradabilityRow({ cell }: { cell: TradabilityCell }) {
  return (
    <tr className={!cell.tradable ? "flagged" : undefined}>
      <th scope="row" className="mono" style={{ fontWeight: 400 }}>
        {cell.symbol}
      </th>
      <td className="mono">{cell.timeframe}</td>
      <td className="num">{num(cell.median_spread_points, 1)}</td>
      <td className="num">{num(cell.median_atr_points, 1)}</td>
      <td className="num">
        <span className={!cell.tradable ? "neg" : undefined}>
          {pct(cell.spread_atr_ratio, 1)}
        </span>
      </td>
      <td className="num">{pct(cell.spread_stop_share, 1)}</td>
      <td>
        {!cell.judged ? (
          <Badge kind="mute">not judged</Badge>
        ) : cell.tradable ? (
          <Badge kind="ok">testable</Badge>
        ) : (
          <Badge kind="bad">refused</Badge>
        )}
      </td>
      {/* the ratio columns already carry the numbers: this says only what
          they cannot, and the full sentence stays available on hover */}
      <td title={cell.reason} style={{ color: "var(--ink-faint)", fontSize: 12 }}>
        {note(cell)}
      </td>
    </tr>
  );
}
