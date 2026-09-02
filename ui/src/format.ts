/**
 * Formatting, not computation.
 *
 * No metric is derived here: percentages, deltas and drawdowns arrive
 * already computed from the API. This file only decides how many digits to
 * show and how to write a date.
 *
 * Every timestamp is UTC and is written as such: the MT5 server clock
 * (Europe/Athens) is neither local time nor UTC, and confusing them on an
 * intraday backtest means reading the trades at the wrong hour.
 */

const NBSP = "\u00a0";
const LOCALE = "en-US";

export function num(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString(LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toLocaleString(LOCALE, { maximumFractionDigits: 0 });
}

export function pct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toLocaleString(LOCALE, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}%`;
}

export function signedPct(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${pct(Math.abs(value), digits)}`;
}

export function money(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return num(value, digits);
}

export function signedMoney(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${num(Math.abs(value), digits)}`;
}

/** Format chosen by the API for a row of the comparison table. */
export function byFormat(
  value: number | null | undefined,
  format: "pct" | "money" | "number" | "int" | "text",
  signed = false,
): string {
  switch (format) {
    case "pct":
      return signed ? signedPct(value) : pct(value);
    case "money":
      return signed ? signedMoney(value) : money(value);
    case "int":
      return signed ? signedMoney(value, 0) : int(value);
    default:
      return signed ? signedMoney(value, 3) : num(value, 3);
  }
}

function pad(value: number): string {
  return value.toString().padStart(2, "0");
}

/** Date and time in UTC, explicitly. Never the browser's local time. */
export function utcDateTime(iso: string | null | undefined, withSeconds = false): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  const time =
    `${pad(date.getUTCHours())}:${pad(date.getUTCMinutes())}` +
    (withSeconds ? `:${pad(date.getUTCSeconds())}` : "");
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(
    date.getUTCDate(),
  )}${NBSP}${time}`;
}

export function utcDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}`;
}

/** `2025-04-02` for date inputs, which know nothing about time zones. */
export function isoDateInput(iso: string | null | undefined): string {
  return iso ? utcDate(iso) : "";
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${num(seconds, 1)}${NBSP}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}${NBSP}min ${Math.round(seconds % 60)}${NBSP}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}${NBSP}h ${minutes % 60}${NBSP}min`;
  return `${Math.floor(hours / 24)}${NBSP}d ${hours % 24}${NBSP}h`;
}

/** Sign of a quantity, so that colour is never the only carrier of meaning. */
export function tone(value: number | null | undefined): "pos" | "neg" | "flat" {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) {
    return "flat";
  }
  return value > 0 ? "pos" : "neg";
}
