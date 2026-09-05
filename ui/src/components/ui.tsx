/** Interface pieces reused by the pages. No computation in here. */
import type { ReactNode } from "react";
import { ApiError } from "../api/client";
import { tone } from "../format";

export function Panel(props: {
  title?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  tight?: boolean;
  /** `reveal` on the panels that carry a verdict. Nothing else uses it. */
  className?: string;
}) {
  return (
    <section className={`panel${props.className ? ` ${props.className}` : ""}`}>
      {(props.title || props.aside) && (
        <header className="panel-head">
          {typeof props.title === "string" ? <h2>{props.title}</h2> : props.title}
          {props.aside}
        </header>
      )}
      <div className={`panel-body${props.tight ? " tight" : ""}`}>{props.children}</div>
    </section>
  );
}

export function Notice(props: {
  kind: "error" | "warn" | "ok" | "info";
  title?: string;
  children: ReactNode;
}) {
  return (
    <div className={`notice ${props.kind}`} role={props.kind === "error" ? "alert" : undefined}>
      {props.title && <strong>{props.title}</strong>}
      {props.children}
    </div>
  );
}

/** The error with its remedy, not just the symptom. */
export function ErrorNotice({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error);
  const status = error instanceof ApiError ? error.status : undefined;
  const remedy =
    status === 409
      ? "Download the data first with `python -m examples.download_year <SYMBOL>`."
      : status === 503
        ? "Start MetaTrader 5, leave it connected, then retry."
        : status === 422
          ? "Fix the strategy spec and reload."
          : status === 0
            ? "Restart the backend with `python run.py`."
            : "The full detail is in the server log.";
  return (
    <Notice kind="error" title={status ? `Error ${status}` : "Error"}>
      <div>{message}</div>
      <div style={{ marginTop: 4, color: "var(--ink-soft)" }}>{remedy}</div>
    </Notice>
  );
}

/** Skeleton with reserved height: the layout does not jump when data lands. */
export function Skeleton({ height = 18, width = "100%" }: { height?: number; width?: string }) {
  return <div className="skeleton" style={{ height, width }} aria-hidden="true" />;
}

export function Loading({ label, height = 180 }: { label: string; height?: number }) {
  return (
    <div aria-live="polite">
      <div style={{ color: "var(--ink-soft)", marginBottom: 8 }}>{label}</div>
      <Skeleton height={height} />
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <p>
        <strong>{title}</strong>
      </p>
      {children}
    </div>
  );
}

/** Value with an explicit sign: colour is not the only carrier of meaning. */
export function Signed({ value, text }: { value: number | null | undefined; text: string }) {
  return <span className={tone(value)}>{text}</span>;
}

export function Badge({
  kind,
  children,
}: {
  kind: "ok" | "bad" | "warn" | "info" | "mute";
  children: ReactNode;
}) {
  return <span className={`badge ${kind}`}>{children}</span>;
}

export function StatusBadge({ status }: { status: "running" | "done" | "error" }) {
  const map = {
    running: { kind: "info", label: "running" },
    done: { kind: "ok", label: "done" },
    error: { kind: "bad", label: "failed" },
  } as const;
  const entry = map[status];
  return <Badge kind={entry.kind}>{entry.label}</Badge>;
}

export function Field(props: {
  label: string;
  hint?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label htmlFor={props.htmlFor}>{props.label}</label>
      {props.children}
      <span className="hint">{props.hint}</span>
    </div>
  );
}
