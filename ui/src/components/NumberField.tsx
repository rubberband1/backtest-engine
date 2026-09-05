import { useEffect, useRef, useState } from "react";

/**
 * A numeric input that always uses a dot for the decimal separator.
 *
 * `<input type="number">` renders and parses according to the *browser's*
 * locale, which no attribute overrides - not `lang`, not the document's. On a
 * machine set to Italian it showed `0,4` in the field while the JSON panel
 * beside it showed `0.4`, for the same value, in an interface that is
 * otherwise entirely in English. Two notations for one number, on screen at
 * the same time, is a correctness problem before it is a cosmetic one: the
 * field is what the user edits and the JSON is what the engine runs.
 *
 * So the field is a text input that does its own parsing. `inputMode`
 * "decimal" still brings up the numeric keypad on a phone, and a comma typed
 * out of habit is accepted and normalised rather than rejected.
 *
 * The draft is kept as a string while the field has focus, because a number
 * being typed passes through states that are not numbers - "", "-", "0." -
 * and re-rendering those from a parsed value deletes the character the user
 * just typed.
 */
export type NumberFieldProps = {
  id?: string;
  value: number | null | undefined;
  /** `null` when the field is empty. */
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  /** Decimal places to allow. Omit for unrestricted. */
  step?: number | "any";
  disabled?: boolean;
  placeholder?: string;
  className?: string;
  "aria-label"?: string;
  "aria-describedby"?: string;
};

function format(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  // toString gives "0.4", never "0,4", whatever the machine's locale
  return String(value);
}

export function NumberField({
  id,
  value,
  onChange,
  min,
  max,
  step,
  disabled,
  placeholder,
  className,
  ...aria
}: NumberFieldProps) {
  const [draft, setDraft] = useState(() => format(value));
  const editing = useRef(false);

  // follow the value when it changes underneath us (a preset applied, a spec
  // loaded), but never while the user is mid-edit
  useEffect(() => {
    if (!editing.current) setDraft(format(value));
  }, [value]);

  function commit(text: string) {
    setDraft(text);
    const normalised = text.replace(",", ".").trim();
    if (normalised === "") {
      onChange(null);
      return;
    }
    const parsed = Number(normalised);
    // "-" and "1e" are on the way to a number: keep them in the draft and do
    // not report a value yet
    if (!Number.isFinite(parsed)) return;
    onChange(parsed);
  }

  function clamp() {
    editing.current = false;
    const normalised = draft.replace(",", ".").trim();
    if (normalised === "") {
      setDraft("");
      onChange(null);
      return;
    }
    let parsed = Number(normalised);
    if (!Number.isFinite(parsed)) {
      setDraft(format(value));
      return;
    }
    if (min !== undefined && parsed < min) parsed = min;
    if (max !== undefined && parsed > max) parsed = max;
    setDraft(format(parsed));
    onChange(parsed);
  }

  return (
    <input
      {...aria}
      id={id}
      type="text"
      inputMode="decimal"
      autoComplete="off"
      spellCheck={false}
      className={className}
      disabled={disabled}
      placeholder={placeholder}
      value={draft}
      onFocus={() => {
        editing.current = true;
      }}
      onChange={(event) => commit(event.target.value)}
      onBlur={clamp}
    />
  );
}
