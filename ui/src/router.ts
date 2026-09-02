/** Minimal hash router: five pages do not justify a dependency. */
import { useEffect, useState } from "react";

export type Route =
  | { name: "run" }
  | { name: "result"; runId: string }
  | { name: "compare"; runIds: string[] }
  | { name: "validation"; runId: string }
  | { name: "batch" };

export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#/, "");
  const [raw, query = ""] = path.split("?");
  const parts = (raw ?? "").split("/").filter(Boolean);

  if (parts[0] === "result") return { name: "result", runId: parts[1] ?? "" };
  if (parts[0] === "validation") return { name: "validation", runId: parts[1] ?? "" };
  if (parts[0] === "batch") return { name: "batch" };
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
