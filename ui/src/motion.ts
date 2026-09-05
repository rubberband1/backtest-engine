/**
 * The two questions a component has to ask before it animates.
 *
 * The rule this file exists to enforce: a curve is drawn once, when it first
 * arrives, and never again. A chart that redraws itself every time a filter
 * changes is a chart nobody can read while they are working.
 */
import { useEffect, useRef, useState } from "react";

export function prefersReducedMotion(): boolean {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/**
 * True on the first render for a given key, false afterwards.
 *
 * Pass the run id, the session id - whatever identifies the series. Charts
 * hand the result to Recharts' `isAnimationActive`, so a new run draws and a
 * resize, a tooltip or a re-render does not. Always false when the reader
 * has asked for reduced motion.
 */
export function useFirstDraw(key: string | null | undefined): boolean {
  const drawn = useRef<Set<string>>(new Set());
  const [, force] = useState(0);
  const id = key ?? "";
  const first = id !== "" && !drawn.current.has(id) && !prefersReducedMotion();

  useEffect(() => {
    if (id !== "" && !drawn.current.has(id)) {
      drawn.current.add(id);
      // one re-render after the draw has been handed out, so the next one
      // (a resize, a hover) gets `false` instead of animating again
      const timer = window.setTimeout(() => force((n) => n + 1), DRAW_MS + 60);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [id]);

  return first;
}

/** Kept in step with `--dur-draw` in styles.css. */
export const DRAW_MS = 500;
