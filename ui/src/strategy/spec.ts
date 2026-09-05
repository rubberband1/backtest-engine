/**
 * The shape of a strategy spec, in the editor's terms.
 *
 * These types describe the JSON the engine already validates; they are not a
 * second source of truth. Every list of allowed values - indicators, their
 * parameters, the operators, the bar fields, the features - comes from
 * `/api/vocabulary` at runtime, so adding an indicator to the registry adds
 * it to the editor and nothing here has to be remembered.
 *
 * What does live here is the *shape*: which keys a condition has, what an
 * operand looks like, and what a sensible empty strategy is. Getting that
 * wrong produces a spec the API rejects with a readable message, which is
 * the failure mode this file is allowed to have.
 */

export type Operand =
  | { ref: string }
  | { const: number }
  | { bar: string }
  | { feature: string };

export type OperandKind = "ref" | "const" | "bar" | "feature";

export type CompareOp =
  | "gt"
  | "gte"
  | "lt"
  | "lte"
  | "eq"
  | "cross_above"
  | "cross_below";

export type Condition =
  | { op: "and" | "or"; operands: Condition[] }
  | { op: "not"; operand: Condition }
  | { op: CompareOp; left: Operand; right: Operand }
  | { op: "between"; left: Operand; low: Operand; high: Operand }
  | { op: "rising" | "falling"; operand: Operand; periods: number };

export type Level =
  | { type: "points"; value: number }
  | { type: "percent"; value: number }
  | { type: "atr"; indicator: string; mult: number };

export interface IndicatorEntry {
  id: string;
  type: string;
  params: Record<string, unknown>;
}

export interface Session {
  start: string;
  end: string;
  timezone: "server" | "utc";
}

export interface Spec {
  schema_version: number;
  id: string;
  name: string;
  description: string;
  instrument: { symbol: string; timeframe: string };
  indicators: IndicatorEntry[];
  entry: { long: Condition | null; short: Condition | null };
  exit: {
    stop_loss: Level | null;
    take_profit: Level | null;
    time_stop: { bars: number } | null;
    signal_exit: Condition | null;
  };
  sizing: {
    type: string;
    equity_per_001_lot: number;
    min_lot: number;
    max_lot: number;
  };
  risk: {
    max_open_positions: number;
    cooldown_minutes: number;
    max_trades_per_day: number | null;
    max_spread_points: number | null;
    session: Session | null;
    news_filter: null;
  };
}

export const OPERAND_KINDS: { kind: OperandKind; label: string; hint: string }[] = [
  { kind: "ref", label: "indicator", hint: "a value computed by one of the indicators above" },
  { kind: "bar", label: "bar field", hint: "a price or field of the bar itself" },
  { kind: "feature", label: "feature", hint: "a ratio derived from the candle's geometry" },
  { kind: "const", label: "constant", hint: "a fixed number" },
];

export const COMPARE_LABELS: Record<string, string> = {
  gt: "is above",
  gte: "is at or above",
  lt: "is below",
  lte: "is at or below",
  eq: "equals",
  cross_above: "crosses above",
  cross_below: "crosses below",
  between: "is between",
  rising: "is rising over",
  falling: "is falling over",
  and: "all of",
  or: "any of",
  not: "not",
};

export function operandKind(operand: Operand): OperandKind {
  if ("ref" in operand) return "ref";
  if ("bar" in operand) return "bar";
  if ("feature" in operand) return "feature";
  return "const";
}

export function operandValue(operand: Operand): string {
  if ("ref" in operand) return operand.ref;
  if ("bar" in operand) return operand.bar;
  if ("feature" in operand) return operand.feature;
  return String(operand.const);
}

export function operandLabel(operand: Operand): string {
  if ("const" in operand) return String(operand.const);
  return operandValue(operand);
}

export function emptyOperand(kind: OperandKind, first?: string): Operand {
  switch (kind) {
    case "ref":
      return { ref: first ?? "" };
    case "bar":
      return { bar: first ?? "close" };
    case "feature":
      return { feature: first ?? "" };
    default:
      return { const: 0 };
  }
}

export function emptyCondition(): Condition {
  return { op: "gt", left: { bar: "close" }, right: { const: 0 } };
}

export function isGroup(condition: Condition): condition is {
  op: "and" | "or";
  operands: Condition[];
} {
  return condition.op === "and" || condition.op === "or";
}

export function isNot(condition: Condition): condition is {
  op: "not";
  operand: Condition;
} {
  return condition.op === "not";
}

export function isBetween(condition: Condition): condition is {
  op: "between";
  left: Operand;
  low: Operand;
  high: Operand;
} {
  return condition.op === "between";
}

export function isTrend(condition: Condition): condition is {
  op: "rising" | "falling";
  operand: Operand;
  periods: number;
} {
  return condition.op === "rising" || condition.op === "falling";
}

export function isCompare(condition: Condition): condition is {
  op: CompareOp;
  left: Operand;
  right: Operand;
} {
  return !isGroup(condition) && !isNot(condition) && !isBetween(condition) && !isTrend(condition);
}

/** A spec that is already valid, so the editor never starts on an error. */
export function blankSpec(symbol: string, timeframe: string): Spec {
  return {
    schema_version: 1,
    id: "new-strategy",
    name: "New strategy",
    description: "",
    instrument: { symbol, timeframe },
    indicators: [{ id: "rsi", type: "rsi", params: { period: 14, source: "close" } }],
    entry: {
      long: {
        op: "lt",
        left: { ref: "rsi" },
        right: { const: 30 },
      },
      short: null,
    },
    exit: {
      stop_loss: { type: "points", value: 150 },
      take_profit: { type: "points", value: 100 },
      time_stop: { bars: 60 },
      signal_exit: null,
    },
    sizing: {
      type: "equity_per_step",
      equity_per_001_lot: 100,
      min_lot: 0.01,
      max_lot: 0.05,
    },
    risk: {
      max_open_positions: 1,
      cooldown_minutes: 0,
      max_trades_per_day: null,
      max_spread_points: null,
      session: null,
      news_filter: null,
    },
  };
}

/**
 * Validation errors keyed by the spec path that caused them.
 *
 * The API writes them as `entry.long.operands.0.left: <message>`, which is a
 * path into the JSON. Splitting it here is what lets the message appear
 * under the control that produced it instead of in a list at the bottom that
 * nobody reads.
 */
export function errorsByPath(errors: string[]): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const entry of errors) {
    const separator = entry.indexOf(": ");
    const path = separator > 0 ? entry.slice(0, separator) : "<root>";
    const message = separator > 0 ? entry.slice(separator + 2) : entry;
    out.set(path, [...(out.get(path) ?? []), message]);
  }
  return out;
}

/** Every message whose path starts with `prefix`, deduplicated. */
export function errorsFor(byPath: Map<string, string[]>, prefix: string): string[] {
  const out: string[] = [];
  for (const [path, messages] of byPath) {
    if (path === prefix || path.startsWith(`${prefix}.`)) {
      for (const message of messages) {
        if (!out.includes(message)) out.push(message);
      }
    }
  }
  return out;
}

/** A deep copy that does not go through the structuredClone availability dance. */
export function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}
