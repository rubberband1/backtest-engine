/**
 * Filling the cache from the broker, as a command rather than a side effect.
 *
 * Until now the only way to extend an instrument's history was a script, and
 * the only sign it had worked was that the coverage line silently grew. The
 * download is one of the two things this application does that involve the
 * outside world, and it had no button.
 *
 * Only the holes are fetched, and the panel says so: asking for a period the
 * cache already holds is a valid request that downloads nothing and reports
 * that, which is both the honest answer and the fast one.
 */
import { useEffect, useRef, useState } from "react";
import { api, type Coverage, type DownloadJob } from "../api/client";
import { int, isoDateInput, utcDate } from "../format";
import { Progress } from "./Progress";
import { DownloadIcon } from "./icons";
import { ErrorNotice, Field, Notice } from "./ui";

const POLL_MS = 800;

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

/** A year back from today, the default when nothing is cached at all. */
function aYearAgo(): string {
  const date = new Date();
  date.setUTCFullYear(date.getUTCFullYear() - 1);
  return date.toISOString().slice(0, 10);
}

export function DataUpdate({
  symbol,
  timeframe,
  coverage,
  onFinished,
}: {
  symbol: string;
  timeframe: string;
  coverage: Coverage | null;
  onFinished: () => void;
}) {
  const cached = (coverage?.bars ?? 0) > 0;
  const [start, setStart] = useState("");
  const [end, setEnd] = useState(today);
  const [job, setJob] = useState<DownloadJob | null>(null);
  const [error, setError] = useState<unknown>(null);
  const finished = useRef(onFinished);
  finished.current = onFinished;

  // The default period is the gap the cache does not cover: from the last
  // bar it holds to now. That is the request somebody almost always means,
  // and offering it prefilled is most of what makes this a command rather
  // than a form.
  useEffect(() => {
    setStart(coverage?.end ? isoDateInput(coverage.end) : aYearAgo());
    setEnd(today());
    setJob(null);
    setError(null);
  }, [symbol, timeframe, coverage?.end]);

  const running = job?.status === "running";

  useEffect(() => {
    if (!running || !job) return undefined;
    const jobId = job.job_id;
    let alive = true;
    const timer = window.setInterval(() => {
      api
        .downloadJob(jobId)
        .then((next) => {
          if (!alive) return;
          setJob(next);
          if (next.status !== "running") finished.current();
        })
        .catch((problem) => {
          if (!alive) return;
          setError(problem);
          setJob(null);
        });
    }, POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [running, job]);

  async function download() {
    setError(null);
    try {
      setJob(
        await api.download({
          symbol,
          timeframe,
          start: `${start}T00:00:00Z`,
          end: `${end}T00:00:00Z`,
        }),
      );
    } catch (problem) {
      setError(problem);
    }
  }

  const added = job ? job.bars_after - job.bars_before : 0;

  return (
    <div className="stack">
      <div className="fetch">
        <div className="fetch-summary">
          <div style={{ fontSize: 12, color: "var(--ink-soft)", marginBottom: 2 }}>
            {cached ? "Cached now" : "Nothing cached for this pair"}
          </div>
          <div className="fetch-window">
            {cached
              ? `${int(coverage?.bars)} ${timeframe} bars · ${utcDate(coverage?.start)} → ${utcDate(coverage?.end)}`
              : "—"}
          </div>
        </div>
        <Field label="Fetch from" htmlFor="fetch-start">
          <input
            id="fetch-start"
            type="date"
            value={start}
            max={end}
            disabled={running}
            onChange={(event) => setStart(event.target.value)}
          />
        </Field>
        <Field label="Fetch to" htmlFor="fetch-end">
          <input
            id="fetch-end"
            type="date"
            value={end}
            min={start}
            disabled={running}
            onChange={(event) => setEnd(event.target.value)}
          />
        </Field>
      </div>

      <div className="actions">
        <button
          className="primary"
          onClick={download}
          disabled={running || !start || !end || start >= end}
        >
          <DownloadIcon />
          <span style={{ marginLeft: 6 }}>
            {running ? "Downloading…" : `Download ${symbol} ${timeframe} from MT5`}
          </span>
        </button>
        <span style={{ color: "var(--ink-faint)", fontSize: 12 }}>
          Only the part the cache does not already hold is requested.
        </span>
      </div>

      {error !== null && <ErrorNotice error={error} />}

      {running && (
        <Progress
          label="downloading interval"
          detail={job?.current ?? "asking the terminal what it has"}
          completed={job?.completed_holes}
          total={job?.total_holes}
        />
      )}

      {job?.status === "error" && (
        <Notice kind="error" title="The download stopped">
          <div>{job.error}</div>
          {/* The line above is the server's own sentence and is the one worth
              reading. This one lists the two things that are usually behind
              it, without pretending to know which - the terminal being down is
              the common cause, but over a weekend a running terminal refuses
              too, because it cannot pin the broker's clock with no recent
              tick. */}
          <div style={{ marginTop: 4, color: "var(--ink-soft)" }}>
            MetaTrader 5 has to be running, logged in and connected. It also
            refuses while the market is closed, when it cannot read the
            server's clock from a recent tick. Whatever arrived before the
            failure is in the cache and is not lost.
          </div>
        </Notice>
      )}

      {job?.status === "done" && (
        <Notice
          kind={added > 0 ? "ok" : "info"}
          title={
            added > 0
              ? `Added ${int(added)} bars`
              : "Nothing to download: the cache already covers this period"
          }
        >
          {added > 0 ? (
            <div className="fetch-window">
              {int(job.bars_after)} {job.timeframe} bars now · {utcDate(job.first_bar)} →{" "}
              {utcDate(job.last_bar)} · filled {int(job.completed_holes)} interval
              {job.completed_holes === 1 ? "" : "s"}
            </div>
          ) : (
            <div style={{ color: "var(--ink-soft)" }}>
              Widen the period, or pick a different instrument or timeframe.
            </div>
          )}
        </Notice>
      )}
    </div>
  );
}
