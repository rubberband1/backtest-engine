/**
 * The live runner, visible from every page.
 *
 * A runner that is holding a position is the most consequential thing this
 * application can be doing, and until now the only place that said so was the
 * page about it. Somebody editing a strategy had no way to know. This sits
 * under the header on every screen and is the one element that renders
 * whether or not it has anything to report - a strip that appears only when
 * something is live is a strip nobody learns to look at.
 *
 * It reads diaries through the API. The backend never talks to MT5 about a
 * live session, so opening a page cannot disturb a runner that is trading.
 */
import { useEffect, useState } from "react";
import { api, type LiveSession } from "../api/client";
import { utcDateTime } from "../format";

// The runner works on closed bars, the fastest of which is a minute. Ten
// seconds is well inside that and costs one directory read.
const POLL_MS = 10_000;

type State = {
  kind: "none" | "stopped" | "listening" | "holding";
  session: LiveSession | null;
  errors: number;
};

function summarize(sessions: LiveSession[]): State {
  if (sessions.length === 0) return { kind: "none", session: null, errors: 0 };
  // a runner that is alive outranks one that has stopped, and among live
  // ones the one holding a position is the one worth naming
  const live = sessions.filter((session) => session.running);
  const holding = live.find((session) => session.in_position);
  if (holding) return { kind: "holding", session: holding, errors: holding.errors };
  if (live.length > 0) {
    const first = live[0]!;
    return { kind: "listening", session: first, errors: first.errors };
  }
  const last = sessions[0]!;
  return { kind: "stopped", session: last, errors: last.errors };
}

const WORDS: Record<State["kind"], string> = {
  none: "No live runner",
  stopped: "Live runner stopped",
  listening: "Listening",
  holding: "In position",
};

const DOTS: Record<State["kind"], string> = {
  none: "idle",
  stopped: "idle",
  listening: "listening",
  holding: "holding",
};

export function LiveStrip() {
  const [state, setState] = useState<State | null>(null);

  useEffect(() => {
    let alive = true;
    const read = () => {
      api
        .liveSessions()
        .then((sessions) => alive && setState(summarize(sessions)))
        // the backend being unreachable is already said, loudly, by the page
        // itself: this strip does not add a second alarm about it
        .catch(() => alive && setState(null));
    };
    read();
    const timer = window.setInterval(read, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  const kind = state?.kind ?? "none";
  const session = state?.session ?? null;
  const attention = (state?.errors ?? 0) > 0;

  return (
    <div className={`live-strip${attention ? " attention" : ""}`} role="status">
      <span className="state">
        <span className={`live-dot ${DOTS[kind]}`} aria-hidden="true" />
        {WORDS[kind]}
      </span>
      {session && (
        <>
          <span className="what">
            {session.symbol} {session.timeframe}
            {session.strategy_id ? ` · ${session.strategy_id}` : ""}
          </span>
          {session.dry_run && <span>dry run</span>}
          {attention && (
            <span>
              {state?.errors} error{state?.errors === 1 ? "" : "s"} in the diary
            </span>
          )}
          <span className="spread">
            {/* MT5 server time is neither local time nor UTC: say which. A
                session that has processed no bar yet says that, rather than
                printing an em dash where a timestamp should be. */}
            {session.last_bar
              ? `last bar ${utcDateTime(session.last_bar)} UTC`
              : "no bar processed yet"}{" "}
            · <a href={`#/live/${encodeURIComponent(session.session_id)}`}>open</a>
          </span>
        </>
      )}
      {!session && (
        <span className="spread">
          <a href="#/live">Live</a>
        </span>
      )}
    </div>
  );
}
