/** Minimal hash router: eight pages do not justify a dependency. */
import { useEffect, useState } from "react";

export type Route =
  | { name: "run" }
  | { name: "result"; runId: string }
  | { name: "compare"; runIds: string[] }
  | { name: "validation"; runId: string }
  | { name: "strategy"; strategyId: string | null }
  | { name: "batch" }
  | { name: "screen"; jobId: string | null }
  | { name: "live"; sessionId: string | null };

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, "");
  const [raw, query = ""] = path.split("?");
  const parts = (raw ?? "").split("/").filter(Boolean);

  if (parts[0] === "result") return { name: "result", runId: parts[1] ?? "" };
  if (parts[0] === "validation") return { name: "validation", runId: parts[1] ?? "" };
  if (parts[0] === "batch") return { name: "batch" };
  // the id in the URL is what makes "duplicate and modify" a link rather than
  // a sequence of clicks somebody has to remember
  if (parts[0] === "strategy") {
    return { name: "strategy", strategyId: parts[1] ? decodeURIComponent(parts[1]) : null };
  }
  if (parts[0] === "live") return { name: "live", sessionId: parts[1] ?? null };
  // a campaign runs for minutes: its id lives in the URL so a reload, or a
  // link sent to someone else, reattaches to the job instead of losing it
  if (parts[0] === "screen") {
    return { name: "screen", jobId: new URLSearchParams(query).get("job") };
  }
  if (parts[0] === "compare") {
    const runs = new URLSearchParams(query).get("runs");
    return { name: "compare", runIds: runs ? runs.split(",").filter(Boolean) : [] };
  }
  return { name: "run" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parseRoute(window.location.hash));
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  return route;
}

export function navigate(to: string): void {
  window.location.hash = to;
}
