/**
 * Watching a blocking call while it blocks.
 *
 * The research endpoints answer once, with a report. What they now also do is
 * accept a token and publish their position under it, so the page can poll
 * `/api/progress/{token}` beside the request it is already awaiting and say
 * "permutation 340 / 2000" instead of turning a spinner.
 *
 * The token is minted here, per call. Nothing is retried and nothing is
 * cached: if the poll fails the counter simply stops moving, and the call it
 * describes is unaffected.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Progress } from "./api/client";

const POLL_MS = 500;

function newToken(): string {
  return `w${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export type Watched = {
  /** The last position reported, or null when nothing is running. */
  progress: Progress | null;
  /** Runs `call` with a fresh token and polls it until the call settles. */
  watch: <T>(call: (token: string) => Promise<T>) => Promise<T>;
};

export function useWatched(): Watched {
  const [progress, setProgress] = useState<Progress | null>(null);
  const timer = useRef<number | null>(null);

  const stop = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  // an unmounting page must not leave an interval polling a token nobody
  // will read
  useEffect(() => stop, [stop]);

  const watch = useCallback(
    async <T,>(call: (token: string) => Promise<T>): Promise<T> => {
      const token = newToken();
      stop();
      setProgress(null);
      timer.current = window.setInterval(() => {
        api
          .progress(token)
          .then((state) => setProgress(state))
          .catch(() => {
            /* the counter stops moving; the call it describes does not */
          });
      }, POLL_MS);
      try {
        return await call(token);
      } finally {
        stop();
        setProgress(null);
      }
    },
    [stop],
  );

  return { progress, watch };
}

/** The line under the spinner: what is happening, in the job's own words. */
export function progressDetail(progress: Progress | null): string {
  if (!progress) return "";
  if (progress.current) return progress.current;
  return progress.label;
}
