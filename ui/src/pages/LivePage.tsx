import { useEffect, useState } from "react";
import {
  api,
  type LiveComparison,
  type LiveDetail,
  type LiveEvent,
  type LiveSession,
} from "../api/client";
import { Badge, Empty, ErrorNotice, Loading, Notice, Panel, Signed } from "../components/ui";
import { int, num, signedMoney, utcDateTime } from "../format";

const POLL_MS = 5000;
// A runner on H1 writes one line an hour. Anything older than this and the
// page is looking at a record, not at a live system, and says so.
const STALE_AFTER_MS = 15 * 60 * 1000;

/**
 * What the runner is doing, and what it costs against the simulation.
 *
 * The backend does not own the runner: `scripts.run_live` does, in its own
 * process, holding its own lock. This page reads the diary that process
 * writes, so opening and closing it cannot disturb a strategy that is
 * trading.
 *
 * The panel that matters is the last one. Everything else here is status;
 * expected-versus-realized is the measurement the whole project was built to
 * make, and it is the one number that says what reality costs on top of the
 * simulation.
 */
export function LivePage({ sessionId }: { sessionId: string | null }) {
  const [sessions, setSessions] = useState<LiveSession[] | null>(null);
  const [detail, setDetail] = useState<LiveDetail | null>(null);
  const [comparison, setComparison] = useState<LiveComparison | null>(null);
  const [comparisonError, setComparisonError] = useState<unknown>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let alive = true;
    const load = () => {
      api
        .liveSessions()
        .then((next) => alive && setSessions(next))
        .catch((reason) => alive && setError(reason));
    };
    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setDetail(null);
      setComparison(null);
      return;
    }
    let alive = true;
    setDetail(null);
    setComparison(null);
    setComparisonError(null);

    const load = () => {
      api
        .liveSession(sessionId, 120)
        .then((next) => alive && setDetail(next))
        .catch((reason) => alive && setError(reason));
    };
    load();
    api
      .liveComparison(sessionId)
      .then((next) => alive && setComparison(next))
      .catch((reason) => alive && setComparisonError(reason));

    const timer = window.setInterval(load, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  if (error !== null && sessions === null) {
    return (
      <div className="page">
        <ErrorNotice error={error} />
      </div>
    );
  }

  if (sessions === null) {
    return (
      <div className="page">
        <Panel title="Live runners">
          <Loading label="Reading the diaries under logs/live." height={120} />
        </Panel>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="page">
        <Empty title="No runner has written a diary yet">
          <p>
            Start one from a terminal. It runs in its own process, on closed bars only, and in
            dry run until you say otherwise:
          </p>
          <p>
            <code>
              python -m scripts.run_live strategies/ma-crossover.json --symbol XAUUSD.r
              --timeframe H1
            </code>
          </p>
          <p>
            Adding <code>--send</code> makes it place orders, and even then it refuses to start
            on anything but a demo account.
          </p>
        </Empty>
      </div>
    );
  }

  const selected = sessionId
    ? (sessions.find((item) => item.session_id === sessionId) ?? detail?.session ?? null)
    : null;

  return (
    <div className="page">
      <Panel title="Live runners" tight>
        <div className="table-scroll">
          <table>
            <caption>
              One row per diary under <code>logs/live</code>. “Running” means a process with
              that lock’s PID is alive right now.
            </caption>
            <thead>
              <tr>
                <th scope="col">Session</th>
                <th scope="col">Strategy</th>
                <th scope="col">Instrument</th>
                <th scope="col">TF</th>
                <th scope="col">Mode</th>
                <th scope="col" className="num">Bars</th>
                <th scope="col" className="num">Trades</th>
                <th scope="col">Last bar (UTC)</th>
                <th scope="col">State</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => (
                <tr
                  key={session.session_id}
                  className={session.session_id === sessionId ? "flagged" : undefined}
                >
                  <th scope="row" style={{ fontWeight: 400 }}>
                    <a href={`#/live/${session.session_id}`} className="mono">
                      {session.session_id}
                    </a>
                  </th>
                  <td className="mono">{session.strategy_id ?? "—"}</td>
                  <td className="mono">{session.symbol || "—"}</td>
                  <td className="mono">{session.timeframe || "—"}</td>
                  <td>
                    {session.dry_run ? (
                      <Badge kind="info">dry run</Badge>
                    ) : (
                      <Badge kind="warn">sending orders</Badge>
                    )}
                  </td>
                  <td className="num">{int(session.bars_processed)}</td>
                  <td className="num">{int(session.trades)}</td>
                  <td className="mono">{utcDateTime(session.last_bar)}</td>
                  <td>
                    <SessionState session={session} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {selected && detail && (
        <>
          <RunnerStatus session={selected} />
          <ComparisonPanel report={comparison} error={comparisonError} />
          <TradesPanel detail={detail} />
          <JournalPanel events={detail.events ?? []} />
        </>
      )}

      {!sessionId && (
        <Notice kind="info" title="Pick a runner">
          Its state, its recent decisions and the expected-versus-realized table appear here.
        </Notice>
      )}
    </div>
  );
}

/** Live, stale or stopped — never “live” for a record that stopped updating. */
function SessionState({ session }: { session: LiveSession }) {
  const age = session.last_event_at
    ? Date.now() - new Date(session.last_event_at).getTime()
    : Number.POSITIVE_INFINITY;

  if (session.errors > 0) return <Badge kind="bad">{int(session.errors)} errors</Badge>;
  if (session.running && age > STALE_AFTER_MS) {
    return <Badge kind="warn">running, no bar for a while</Badge>;
  }
  if (session.running) return <Badge kind="ok">running</Badge>;
  if (session.stopped) return <Badge kind="mute">stopped</Badge>;
  // a lock still on disk with a dead PID is a crash; no lock at all means
  // nothing was ever holding this diary, which is what a replay looks like
  if (session.has_lock) return <Badge kind="bad">process gone, not stopped cleanly</Badge>;
  return <Badge kind="mute">not a live process</Badge>;
}

function RunnerStatus({ session }: { session: LiveSession }) {
  const guard = session.account_guard as Record<string, unknown> | null;
  const mode = guard?.["trade_mode_name"] as string | undefined;
  const reason = guard?.["reason"] as string | undefined;
  const stale =
    session.running &&
    session.last_event_at &&
    Date.now() - new Date(session.last_event_at).getTime() > STALE_AFTER_MS;

  return (
    <Panel
      title={`${session.strategy_id ?? "runner"} · ${session.symbol} ${session.timeframe}`}
      aside={<SessionState session={session} />}
    >
      <div className="stack">
        {session.dry_run ? (
          <Notice kind="info" title="Dry run">
            Nothing was sent to the broker. Every fill below is the price the engine expected,
            not one the market gave, so the comparison measures the engine against itself.
          </Notice>
        ) : (
          <Notice kind="warn" title={`Sending orders on a ${mode ?? "?"} account`}>
            {reason ?? "Orders from this runner reach the broker."}
          </Notice>
        )}

        {stale && (
          <Notice kind="warn" title="This data is not fresh">
            The process is alive but has not written a line since{" "}
            {utcDateTime(session.last_event_at, true)} UTC. Either the market is closed, or the
            runner is not receiving bars.
          </Notice>
        )}

        {!session.running && !session.stopped && session.has_lock && (
          <Notice kind="error" title="The process is gone and did not stop cleanly">
            The lock names a PID that is not running. Whatever position it held is still with the
            broker: check the terminal before starting another runner on this spec.
          </Notice>
        )}

        {!session.running && !session.stopped && !session.has_lock && (
          <Notice kind="info" title="Not a live process">
            No lock file sits beside this diary, so nothing was ever holding it. It was written
            by a replay: the runner consumed historical bars as if they were arriving now, which
            is how the engine is checked against the backtester.
          </Notice>
        )}

        <div className="kpis">
          <div className="kpi">
            <div className="label">Bars processed</div>
            <div className="value">{int(session.bars_processed)}</div>
            <div className="sub">closed bars only, never the forming one</div>
          </div>
          <div className="kpi">
            <div className="label">Trades closed</div>
            <div className="value">{int(session.trades)}</div>
            <div className="sub">as recorded in the diary</div>
          </div>
          <div className="kpi">
            <div className="label">Last bar (UTC)</div>
            <div className="value small mono">{utcDateTime(session.last_bar, true)}</div>
            <div className="sub">
              first {utcDateTime(session.first_bar)} · engine {session.engine_version ?? "—"}
            </div>
          </div>
          <div className="kpi">
            <div className="label">Process</div>
            <div className="value small">{session.pid ? `pid ${int(session.pid)}` : "—"}</div>
            <div className="sub">one runner per spec, enforced by a lock file</div>
          </div>
        </div>
      </div>
    </Panel>
  );
}

/** The deliverable: what the simulation said, and what the market did. */
function ComparisonPanel({
  report,
  error,
}: {
  report: LiveComparison | null;
  error: unknown;
}) {
  if (error !== null) {
    return (
      <Panel title="Expected vs realized">
        <ErrorNotice error={error} />
      </Panel>
    );
  }
  if (!report) {
    return (
      <Panel title="Expected vs realized">
        <Loading label="Backtesting exactly the bars this runner saw." height={140} />
      </Panel>
    );
  }

  const gap = report.realized_pnl - report.expected_pnl;
  const onlyExpected = report.only_expected ?? [];
  const onlyRealized = report.only_realized ?? [];
  const deviations = report.deviations ?? [];
  return (
    <Panel
      title="Expected vs realized"
      aside={
        report.matched === 0 ? (
          <Badge kind="warn">nothing matched</Badge>
        ) : report.pnl_unexplained === 0 ? (
          <Badge kind="ok">fully accounted for</Badge>
        ) : (
          <Badge kind="warn">{signedMoney(report.pnl_unexplained)} unexplained</Badge>
        )
      }
    >
      <div className="stack">
        <Notice kind={report.matched === 0 ? "warn" : "info"} title="Verdict">
          {report.verdict}
        </Notice>

        <div className="kpis">
          <div className="kpi">
            <div className="label">PnL gap</div>
            <div className="value">
              <Signed value={gap} text={signedMoney(gap)} />
            </div>
            <div className="sub">
              {signedMoney(report.realized_pnl)} realized vs {signedMoney(report.expected_pnl)}{" "}
              expected
            </div>
          </div>
          <div className="kpi">
            <div className="label">From slippage</div>
            <div className="value small">
              <Signed value={report.pnl_from_slippage} text={signedMoney(report.pnl_from_slippage)} />
            </div>
            <div className="sub">on the {int(report.matched)} trades both records took</div>
          </div>
          <div className="kpi">
            <div className="label">From unmatched trades</div>
            <div className="value small">
              <Signed value={report.pnl_from_unmatched} text={signedMoney(report.pnl_from_unmatched)} />
            </div>
            <div className="sub">
              {int(onlyExpected.length)} missed · {int(onlyRealized.length)} extra
            </div>
          </div>
          <div className="kpi">
            <div className="label">Unexplained</div>
            <div className="value small">
              <Signed value={report.pnl_unexplained} text={signedMoney(report.pnl_unexplained)} />
            </div>
            <div className="sub">what neither slippage nor unmatched trades account for</div>
          </div>
          <div className="kpi">
            <div className="label">Entry slippage</div>
            <div className="value small">
              {num(report.median_entry_slippage_points, 2)} pt
            </div>
            <div className="sub">
              median · p90 {num(report.p90_entry_slippage_points, 2)} pt
            </div>
          </div>
        </div>

        {(report.rejected_orders > 0 || report.partial_fills > 0) && (
          <Notice kind="warn" title="The broker did not do everything it was asked">
            {int(report.rejected_orders)} order(s) rejected and {int(report.partial_fills)}{" "}
            partially filled. Those are trades the strategy asked for and did not get, which is
            not slippage and is not in the simulation at all.
          </Notice>
        )}

        {(onlyExpected.length > 0 || onlyRealized.length > 0) && (
          <div className="table-scroll">
            <table>
              <caption>Trades only one of the two records holds, with the reason.</caption>
              <thead>
                <tr>
                  <th scope="col">Entry (UTC)</th>
                  <th scope="col">Side</th>
                  <th scope="col" className="num">Net PnL</th>
                  <th scope="col">Where</th>
                  <th scope="col">Why</th>
                </tr>
              </thead>
              <tbody>
                {[...onlyExpected, ...onlyRealized].map((trade) => (
                  <tr key={`${trade.side}-${trade.entry_time}`}>
                    <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                      {utcDateTime(trade.entry_time, true)}
                    </th>
                    <td>{trade.direction > 0 ? "long" : "short"}</td>
                    <td className="num">
                      <Signed value={trade.net_pnl} text={signedMoney(trade.net_pnl)} />
                    </td>
                    <td>
                      <Badge kind={trade.side === "expected" ? "warn" : "info"}>
                        {trade.side === "expected" ? "backtest only" : "live only"}
                      </Badge>
                    </td>
                    <td>{trade.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {deviations.length > 0 && (
          <div className="table-scroll">
            <table>
              <caption>
                Per-trade deviation on the trades both records took. Slippage is signed in the
                direction of the trade: negative is worse than expected.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Entry (UTC)</th>
                  <th scope="col">Side</th>
                  <th scope="col" className="num">Entry slip (pt)</th>
                  <th scope="col" className="num">Lots</th>
                  <th scope="col" className="num">PnL difference</th>
                  <th scope="col">Exit reason</th>
                </tr>
              </thead>
              <tbody>
                {deviations.slice(0, 100).map((deviation) => (
                  <tr key={deviation.entry_time} className={deviation.exit_reason_differs ? "flagged" : undefined}>
                    <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                      {utcDateTime(deviation.entry_time, true)}
                    </th>
                    <td>{deviation.direction > 0 ? "long" : "short"}</td>
                    <td className="num">{num(deviation.entry_slippage_points, 2)}</td>
                    <td className="num">
                      {num(deviation.lots_realized, 2)}
                      {deviation.lots_realized !== deviation.lots_expected
                        ? ` of ${num(deviation.lots_expected, 2)}`
                        : ""}
                    </td>
                    <td className="num">
                      <Signed
                        value={deviation.pnl_difference}
                        text={signedMoney(deviation.pnl_difference)}
                      />
                    </td>
                    <td>
                      {deviation.exit_reason_differs ? (
                        <span className="neg">
                          {deviation.exit_reason_realized} (expected{" "}
                          {deviation.exit_reason_expected})
                        </span>
                      ) : (
                        deviation.exit_reason_realized
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {(report.warnings ?? []).map((warning) => (
          <Notice key={warning} kind="warn" title="Careful">
            {warning}
          </Notice>
        ))}
      </div>
    </Panel>
  );
}

function TradesPanel({ detail }: { detail: LiveDetail }) {
  const trades = detail.trades ?? [];
  if (trades.length === 0) {
    return (
      <Panel title="Trades">
        <Empty title="No position has been closed yet">
          <p>
            The runner has processed {int(detail.session.bars_processed)} closed bars. Until a
            position opens and closes there is nothing to compare.
          </p>
        </Empty>
      </Panel>
    );
  }
  return (
    <Panel title={`Trades (${int(trades.length)})`} tight>
      <div className="table-scroll">
        <table>
          <caption>Closed positions as the runner recorded them, newest last.</caption>
          <thead>
            <tr>
              <th scope="col">Entry (UTC)</th>
              <th scope="col">Exit (UTC)</th>
              <th scope="col">Side</th>
              <th scope="col" className="num">Lots</th>
              <th scope="col" className="num">Entry</th>
              <th scope="col" className="num">Exit</th>
              <th scope="col">Reason</th>
              <th scope="col" className="num">Net PnL</th>
              <th scope="col" className="num">R</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade) => (
              <tr key={`${trade.entry_time}-${trade.exit_time}`}>
                <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                  {utcDateTime(trade.entry_time, true)}
                </th>
                <td className="mono">{utcDateTime(trade.exit_time, true)}</td>
                <td>{trade.direction > 0 ? "long" : "short"}</td>
                <td className="num">{num(trade.lots, 2)}</td>
                <td className="num">{num(trade.entry_price, 5)}</td>
                <td className="num">{num(trade.exit_price, 5)}</td>
                <td>
                  {trade.exit_reason}
                  {trade.ambiguous ? " (ambiguous)" : ""}
                </td>
                <td className="num">
                  <Signed value={trade.net_pnl} text={signedMoney(trade.net_pnl)} />
                </td>
                <td className="num">
                  <Signed value={trade.r_multiple} text={signedMoney(trade.r_multiple, 2)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/** The decisions, not only the fills: a bar with no signal is a decision too. */
function JournalPanel({ events }: { events: LiveEvent[] }) {
  const [showBars, setShowBars] = useState(false);
  const shown = showBars ? events : events.filter((event) => event.kind !== "bar");

  return (
    <Panel
      title="Execution diary"
      aside={
        <label style={{ fontSize: 12, color: "var(--ink-soft)" }}>
          <input
            type="checkbox"
            checked={showBars}
            onChange={() => setShowBars((value) => !value)}
          />{" "}
          include every processed bar
        </label>
      }
      tight
    >
      <div className="table-scroll" style={{ maxHeight: 420 }}>
        <table>
          <caption>
            Newest first. Each line is one decision, with what it was decided on. The file is
            append-only JSON at <code>logs/live</code> and holds no account identifier.
          </caption>
          <thead>
            <tr>
              <th scope="col">At (UTC)</th>
              <th scope="col">Bar (UTC)</th>
              <th scope="col">Event</th>
              <th scope="col">Detail</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((event, index) => (
              <tr key={`${event.at}-${index}`} className={event.kind === "error" ? "flagged" : undefined}>
                <th scope="row" className="mono" style={{ fontWeight: 400 }}>
                  {utcDateTime(event.at, true)}
                </th>
                <td className="mono">{utcDateTime(event.bar_time, true)}</td>
                <td>
                  <Badge kind={badgeFor(event.kind)}>{event.kind.replace(/_/g, " ")}</Badge>
                </td>
                <td className="mono" style={{ fontSize: 12, color: "var(--ink-soft)" }}>
                  {summarize(event)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown.length === 0 && (
        <Empty title="Nothing but bars in this window">
          <p>Tick the box above to see every processed bar and the signal it produced.</p>
        </Empty>
      )}
    </Panel>
  );
}

function badgeFor(kind: string): "ok" | "bad" | "warn" | "info" | "mute" {
  if (kind === "error") return "bad";
  if (kind === "order_sent" || kind === "order_result") return "warn";
  if (kind === "position_opened" || kind === "position_closed") return "ok";
  if (kind === "started" || kind === "reconciled") return "info";
  return "mute";
}

/** One readable line per event: the diary is dense, the table must not be. */
function summarize(event: LiveEvent): string {
  const detail = event.detail as Record<string, any>;
  switch (event.kind) {
    case "started":
      return `${detail.strategy_id ?? "?"} · ${detail.dry_run ? "dry run" : "sending"} · ${int(
        detail.warmup_bars,
      )} bars of history · engine ${detail.engine_version ?? "?"}`;
    case "reconciled":
      return `adopted ${detail.position?.lots ?? "?"} lots opened ${utcDateTime(
        detail.position?.opened_at,
      )}`;
    case "bar": {
      const signal = detail.signal ?? {};
      const side = signal.long ? "long" : signal.short ? "short" : signal.exit ? "exit" : "none";
      return `close ${num(detail.bar?.close, 5)} · spread ${num(
        detail.bar?.spread_points,
        0,
      )} pt · signal ${side}${detail.in_position ? " · in position" : ""}`;
    }
    case "order_sent":
      return `${detail.request?.side ?? "?"} ${num(detail.request?.lots, 2)} lots · expected ${num(
        detail.expected_price,
        5,
      )} · sl ${num(detail.request?.stop_level, 5)} tp ${num(detail.request?.target_level, 5)}`;
    case "order_result": {
      const result = detail.result ?? {};
      return `${result.retcode_name ?? "?"} · filled ${num(result.filled_lots, 2)} of ${num(
        result.requested_lots,
        2,
      )} at ${num(result.filled_price, 5)} (expected ${num(detail.expected_price, 5)})`;
    }
    case "position_closed": {
      const trade = detail.trade ?? {};
      return `${trade.exit_reason ?? "?"} · ${signedMoney(trade.net_pnl)} · R ${signedMoney(
        trade.r_multiple,
        2,
      )}`;
    }
    case "error":
      return String(detail.message ?? detail.reason ?? "");
    case "stopped":
      return `${int(detail.trades)} trades`;
    default:
      return "";
  }
}
