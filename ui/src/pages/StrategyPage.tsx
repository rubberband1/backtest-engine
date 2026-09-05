import { NumberField } from "../components/NumberField";
/**
 * The strategy builder.
 *
 * Until now a strategy was a JSON file written by hand and the UI could only
 * pick one from a menu. This page builds one with controls, validates it
 * against the API as it changes, and - the part that matters - says what it
 * is about to cost *before* anything is run.
 *
 * The feedback panel is not a nicety. An editor makes variants cheap to try,
 * and trying variants is exactly how overfitting is produced; so the page
 * shows, at all times and without being asked, how many attempts are already
 * registered on this instrument and period and what Sharpe a result would
 * have to reach to survive that many. And every run launched from here goes
 * through `/api/backtest` like any other: there is deliberately no path out
 * of this page that produces a result the campaign does not count.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  ApiError,
  type Preview,
  type RunConfigIn,
  type Strategy,
  type SymbolList,
  type ValidateResponse,
  type Vocabulary,
} from "../api/client";
import { ConditionEditor } from "../components/ConditionEditor";
import {
  Badge,
  Empty,
  ErrorNotice,
  Field,
  Loading,
  Notice,
  Panel,
} from "../components/ui";
import { int, isoDateInput, num, pct, signedMoney } from "../format";
import { navigate } from "../router";
import {
  type Condition,
  type IndicatorEntry,
  type Level,
  type Spec,
  blankSpec,
  clone,
  emptyCondition,
  errorsByPath,
  errorsFor,
} from "../strategy/spec";

const VALIDATE_DEBOUNCE_MS = 300;
const PREVIEW_DEBOUNCE_MS = 700;

function toIso(day: string): string | null {
  return day ? `${day}T00:00:00Z` : null;
}

/** A spec parameter's default, from the registry rather than from memory. */
function defaultParams(vocabulary: Vocabulary, type: string): Record<string, unknown> {
  const definition = vocabulary.indicators.find((entry) => entry.name === type);
  const params: Record<string, unknown> = {};
  for (const param of definition?.params ?? []) {
    if (param.default !== null && param.default !== undefined) {
      params[param.name] = param.default;
    }
  }
  return params;
}

export function StrategyPage({ strategyId }: { strategyId: string | null }) {
  const [vocabulary, setVocabulary] = useState<Vocabulary | null>(null);
  const [library, setLibrary] = useState<Strategy[] | null>(null);
  const [environment, setEnvironment] = useState<SymbolList | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);

  const [spec, setSpec] = useState<Spec | null>(null);
  const [validation, setValidation] = useState<ValidateResponse | null>(null);
  const [validating, setValidating] = useState(false);

  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [equity, setEquity] = useState(100);
  const [spreadMode, setSpreadMode] = useState<RunConfigIn["spread_mode"]>("fixed");
  const [spreadValue, setSpreadValue] = useState<number | "">(3);
  const [commission, setCommission] = useState(0);

  const [preview, setPreview] = useState<Preview | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<unknown>(null);
  const [previewStale, setPreviewStale] = useState(false);

  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<unknown>(null);
  const [runError, setRunError] = useState<unknown>(null);
  const [runBusy, setRunBusy] = useState(false);
  const importRef = useRef<HTMLInputElement | null>(null);

  // -- loading -------------------------------------------------------------

  useEffect(() => {
    Promise.all([api.vocabulary(), api.strategies(), api.symbols()])
      .then(([vocab, strategies, symbols]) => {
        setVocabulary(vocab);
        setLibrary(strategies);
        setEnvironment(symbols);
        const source = strategyId
          ? strategies.find((entry) => entry.id === strategyId)
          : undefined;
        setSpec(
          source
            ? duplicateOf(source)
            : blankSpec(symbols.symbols[0]?.name ?? "XAUUSD.r", "H1"),
        );
      })
      .catch(setLoadError);
  }, [strategyId]);

  function duplicateOf(source: Strategy): Spec {
    const copy = clone(source.spec) as unknown as Spec;
    copy.id = `${source.id}-copy`;
    copy.name = `${source.name} (copy)`;
    return copy;
  }

  // The period follows what is actually cached: asking for dates that do not
  // exist is the most common way to get "no data" out of a valid spec.
  //
  // Changing instrument also discards the previous preview instead of marking
  // it stale. A stale number is one edit behind; a number computed on another
  // instrument under this instrument's name is simply wrong, and no banner
  // makes it less so.
  useEffect(() => {
    if (!spec?.instrument.symbol) return;
    let cancelled = false;
    setPreview(null);
    setPreviewError(null);
    api
      .coverage(spec.instrument.symbol, spec.instrument.timeframe)
      .then((coverage) => {
        if (cancelled) return;
        setStart(isoDateInput(coverage.start));
        setEnd(isoDateInput(coverage.end));
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [spec?.instrument.symbol, spec?.instrument.timeframe]);

  // -- validation, as it is typed ------------------------------------------

  useEffect(() => {
    if (!spec) return;
    setValidating(true);
    const timer = window.setTimeout(() => {
      api
        .validate(spec)
        .then(setValidation)
        .catch(() => setValidation(null))
        .finally(() => setValidating(false));
    }, VALIDATE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [spec]);

  const config: RunConfigIn = useMemo(
    () => ({
      symbol: spec?.instrument.symbol ?? "",
      timeframe: spec?.instrument.timeframe ?? "H1",
      start: toIso(start),
      end: toIso(end),
      initial_equity: equity,
      spread_mode: spreadMode,
      spread_value: spreadMode === "per_bar" ? null : spreadValue === "" ? null : spreadValue,
      commission_per_lot_per_side: commission,
      swap_mode: "points",
      session_threshold: 0.5,
      per_bar_spread_quantile: 0.5,
    }),
    [spec, start, end, equity, spreadMode, spreadValue, commission],
  );

  const valid = validation?.valid === true;

  // -- the preview ---------------------------------------------------------

  const runPreview = useCallback(async () => {
    if (!spec || !valid || !config.symbol) return;
    setPreviewBusy(true);
    setPreviewError(null);
    try {
      setPreview(await api.preview({ spec, config }));
      setPreviewStale(false);
    } catch (error) {
      setPreviewError(error);
    } finally {
      setPreviewBusy(false);
    }
  }, [spec, valid, config]);

  useEffect(() => {
    if (!valid) return;
    setPreviewStale(true);
    const timer = window.setTimeout(() => void runPreview(), PREVIEW_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [runPreview, valid]);

  // -- actions -------------------------------------------------------------

  async function save(overwrite: boolean) {
    if (!spec) return;
    setSaveError(null);
    setSaveMessage(null);
    try {
      const response = await api.saveStrategy(spec, overwrite);
      setSaveMessage(response.message);
      setLibrary(await api.strategies());
    } catch (error) {
      setSaveError(error);
    }
  }

  async function launch() {
    if (!spec) return;
    setRunBusy(true);
    setRunError(null);
    try {
      const response = await api.backtest({ spec, config });
      navigate(`/result/${response.run_id}`);
    } catch (error) {
      setRunError(error);
    } finally {
      setRunBusy(false);
    }
  }

  function importJson(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        setSpec(JSON.parse(String(reader.result)) as Spec);
        setSaveMessage(`imported ${file.name}`);
        setSaveError(null);
      } catch (error) {
        setSaveError(new Error(`${file.name} is not valid JSON: ${String(error)}`));
      }
    };
    reader.readAsText(file);
  }

  // -- render --------------------------------------------------------------

  if (loadError) return <ErrorNotice error={loadError} />;
  if (!vocabulary || !spec || !environment) {
    return (
      <Panel title="Strategy">
        <Loading label="Loading the vocabulary the editor builds from" height={220} />
      </Panel>
    );
  }

  const errors = errorsByPath(validation?.valid ? [] : (validation?.errors ?? []));
  const indicatorRefs = spec.indicators.flatMap((indicator) => {
    const definition = vocabulary.indicators.find((entry) => entry.name === indicator.type);
    const outputs = definition?.outputs ?? [];
    return outputs.length
      ? outputs.map((output) => `${indicator.id}.${output}`)
      : [indicator.id];
  });
  const shared = { vocabulary, indicatorRefs, errors };

  function update(mutate: (draft: Spec) => void) {
    setSpec((current) => {
      if (!current) return current;
      const draft = clone(current);
      mutate(draft);
      return draft;
    });
  }

  return (
    <div className="stack">
      <Panel
        title="Strategy"
        aside={
          <div className="actions">
            <input
              ref={importRef}
              type="file"
              accept="application/json,.json"
              hidden
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) importJson(file);
                event.target.value = "";
              }}
            />
            <button type="button" onClick={() => importRef.current?.click()}>
              Import JSON
            </button>
            <button type="button" onClick={() => void save(false)} disabled={!valid}>
              Save to strategies/
            </button>
            <button
              type="button"
              className="primary"
              onClick={() => void launch()}
              disabled={!valid || runBusy}
            >
              {runBusy ? "Starting…" : "Run backtest"}
            </button>
          </div>
        }
      >
        <p className="footnote">
          Every run started here is registered in the campaign like any other
          attempt. The panel on the right says what that already costs.
        </p>
        {saveMessage && (
          <Notice kind="ok" title="Saved">
            {saveMessage}
          </Notice>
        )}
        {saveError instanceof ApiError && saveError.status === 409 ? (
          <Notice kind="warn" title="That file already exists">
            <div>{saveError.message}</div>
            <div className="actions" style={{ marginTop: 8 }}>
              <button type="button" onClick={() => void save(true)}>
                Replace it anyway
              </button>
            </div>
          </Notice>
        ) : saveError ? (
          <ErrorNotice error={saveError} />
        ) : null}
        {runError ? <ErrorNotice error={runError} /> : null}
      </Panel>

      <div className="builder">
        <div className="builder-main stack">
          <Panel title="Start from">
            <div className="form-grid">
              <Field
                label="Duplicate a library strategy"
                hint="The most likely way to build one: copy something that works and change it."
              >
                <select
                  value=""
                  onChange={(event) => {
                    const source = library?.find((entry) => entry.id === event.target.value);
                    if (source) navigate(`/strategy/${encodeURIComponent(source.id)}`);
                  }}
                >
                  <option value="">choose one…</option>
                  {(library ?? []).map((entry) => (
                    <option key={entry.id} value={entry.id}>
                      {entry.name} — {entry.symbol} {entry.timeframe}
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="Or start blank" hint="A minimal spec that already validates.">
                <button
                  type="button"
                  onClick={() =>
                    setSpec(
                      blankSpec(
                        spec.instrument.symbol,
                        spec.instrument.timeframe,
                      ),
                    )
                  }
                >
                  New strategy
                </button>
              </Field>
            </div>
          </Panel>

          <Panel title="Identity and instrument">
            <div className="form-grid">
              <Field
                label="id"
                hint="Becomes the file name in strategies/."
                htmlFor="spec-id"
              >
                <input
                  id="spec-id"
                  value={spec.id}
                  onChange={(event) => update((draft) => void (draft.id = event.target.value))}
                />
                <Messages messages={errorsFor(errors, "id")} />
              </Field>
              <Field label="name" htmlFor="spec-name" hint="Shown in every menu.">
                <input
                  id="spec-name"
                  value={spec.name}
                  onChange={(event) => update((draft) => void (draft.name = event.target.value))}
                />
                <Messages messages={errorsFor(errors, "name")} />
              </Field>
              <Field label="symbol" htmlFor="spec-symbol" hint="From the local cache.">
                <select
                  id="spec-symbol"
                  value={spec.instrument.symbol}
                  onChange={(event) =>
                    update((draft) => void (draft.instrument.symbol = event.target.value))
                  }
                >
                  {environment.symbols.map((entry) => (
                    <option key={entry.name} value={entry.name}>
                      {entry.name}
                    </option>
                  ))}
                </select>
                <Messages messages={errorsFor(errors, "instrument.symbol")} />
              </Field>
              <Field label="timeframe" htmlFor="spec-timeframe" hint="Bars the spec runs on.">
                <select
                  id="spec-timeframe"
                  value={spec.instrument.timeframe}
                  onChange={(event) =>
                    update((draft) => void (draft.instrument.timeframe = event.target.value))
                  }
                >
                  {vocabulary.timeframes.map((entry) => (
                    <option key={entry} value={entry}>
                      {entry}
                    </option>
                  ))}
                </select>
                <Messages messages={errorsFor(errors, "instrument.timeframe")} />
              </Field>
            </div>
            <Field
              label="description"
              htmlFor="spec-description"
              hint="What it is meant to capture, and what is already known about it."
            >
              <textarea
                id="spec-description"
                rows={3}
                value={spec.description}
                onChange={(event) =>
                  update((draft) => void (draft.description = event.target.value))
                }
              />
            </Field>
          </Panel>

          <Panel
            title="Indicators"
            aside={
              <button
                type="button"
                onClick={() =>
                  update((draft) => {
                    const type = vocabulary.indicators[0]?.name ?? "sma";
                    draft.indicators.push({
                      id: uniqueId(type, draft.indicators),
                      type,
                      params: defaultParams(vocabulary, type),
                    });
                  })
                }
              >
                Add indicator
              </button>
            }
          >
            {spec.indicators.length === 0 ? (
              <Empty title="No indicators">
                <p className="footnote">
                  A strategy can compare bar fields and features without one, but
                  most reference at least a moving average or an oscillator.
                </p>
              </Empty>
            ) : (
              <div className="stack">
                {spec.indicators.map((indicator, index) => (
                  <IndicatorRow
                    key={index}
                    indicator={indicator}
                    vocabulary={vocabulary}
                    errors={errorsFor(errors, `indicators.${index}`)}
                    onChange={(next) =>
                      update((draft) => void (draft.indicators[index] = next))
                    }
                    onRemove={() =>
                      update(
                        (draft) =>
                          void (draft.indicators = draft.indicators.filter(
                            (_, i) => i !== index,
                          )),
                      )
                    }
                  />
                ))}
              </div>
            )}
          </Panel>

          <Panel title="Entry conditions">
            <ConditionSlot
              label="Long"
              condition={spec.entry.long}
              path="entry.long"
              shared={shared}
              onChange={(next) => update((draft) => void (draft.entry.long = next))}
            />
            <ConditionSlot
              label="Short"
              condition={spec.entry.short}
              path="entry.short"
              shared={shared}
              onChange={(next) => update((draft) => void (draft.entry.short = next))}
            />
            <Messages messages={errorsFor(errors, "entry")} />
          </Panel>

          <Panel title="Exits">
            <div className="form-grid">
              <LevelEditor
                label="Stop loss"
                level={spec.exit.stop_loss}
                atrIndicators={spec.indicators
                  .filter((entry) => entry.type === "atr")
                  .map((entry) => entry.id)}
                vocabulary={vocabulary}
                errors={errorsFor(errors, "exit.stop_loss")}
                onChange={(next) => update((draft) => void (draft.exit.stop_loss = next))}
              />
              <LevelEditor
                label="Take profit"
                level={spec.exit.take_profit}
                atrIndicators={spec.indicators
                  .filter((entry) => entry.type === "atr")
                  .map((entry) => entry.id)}
                vocabulary={vocabulary}
                errors={errorsFor(errors, "exit.take_profit")}
                onChange={(next) => update((draft) => void (draft.exit.take_profit = next))}
              />
              <Field
                label="Time stop (session bars)"
                htmlFor="time-stop"
                hint="Counted on the server clock, so a DST change does not move it."
              >
                <NumberField
                  id="time-stop"
                  min={0}
                  step={1}
                  value={spec.exit.time_stop?.bars ?? null}
                  placeholder="none"
                  onChange={(value) =>
                    update((draft) => {
                      draft.exit.time_stop =
                        value === null || value <= 0 ? null : { bars: value };
                    })
                  }
                />
                <Messages messages={errorsFor(errors, "exit.time_stop")} />
              </Field>
            </div>
            <ConditionSlot
              label="Signal exit"
              condition={spec.exit.signal_exit}
              path="exit.signal_exit"
              shared={shared}
              onChange={(next) => update((draft) => void (draft.exit.signal_exit = next))}
            />
            <Messages messages={errorsFor(errors, "exit")} />
          </Panel>

          <Panel title="Sizing">
            <div className="form-grid">
              <Field
                label="equity per 0.01 lot"
                htmlFor="sizing-step"
                hint="Account currency of equity that buys one minimum lot step."
              >
                <NumberField
                  id="sizing-step"
                  min={0}
                  step="any"
                  value={spec.sizing.equity_per_001_lot}
                  onChange={(value) =>
                    update((draft) => void (draft.sizing.equity_per_001_lot = value ?? 0))
                  }
                />
              </Field>
              <Field label="min lot" htmlFor="sizing-min" hint="The broker's floor, or above it.">
                <NumberField
                  id="sizing-min"
                  min={0}
                  step="any"
                  value={spec.sizing.min_lot}
                  onChange={(value) => update((draft) => void (draft.sizing.min_lot = value ?? 0))}
                />
              </Field>
              <Field label="max lot" htmlFor="sizing-max" hint="Caps the position however large equity grows.">
                <NumberField
                  id="sizing-max"
                  min={0}
                  step="any"
                  value={spec.sizing.max_lot}
                  onChange={(value) => update((draft) => void (draft.sizing.max_lot = value ?? 0))}
                />
              </Field>
            </div>
            <Messages messages={errorsFor(errors, "sizing")} />
          </Panel>

          <Panel title="Risk gates">
            <div className="form-grid">
              <Field
                label="cooldown (minutes)"
                htmlFor="risk-cooldown"
                hint="Minimum wait after a trade closes."
              >
                <NumberField
                  id="risk-cooldown"
                  min={0}
                  step={1}
                  value={spec.risk.cooldown_minutes}
                  onChange={(value) =>
                    update((draft) => void (draft.risk.cooldown_minutes = value ?? 0))
                  }
                />
              </Field>
              <Field
                label="max trades per day"
                htmlFor="risk-trades"
                hint="Empty means no limit."
              >
                <NumberField
                  id="risk-trades"
                  min={1}
                  step={1}
                  placeholder="no limit"
                  value={spec.risk.max_trades_per_day ?? null}
                  onChange={(value) =>
                    update((draft) => void (draft.risk.max_trades_per_day = value))
                  }
                />
              </Field>
              <Field
                label="max spread (points)"
                htmlFor="risk-spread"
                hint="Entries above this are refused. Empty means no limit."
              >
                <NumberField
                  id="risk-spread"
                  min={0}
                  step="any"
                  placeholder="no limit"
                  value={spec.risk.max_spread_points ?? null}
                  onChange={(value) =>
                    update((draft) => void (draft.risk.max_spread_points = value))
                  }
                />
              </Field>
            </div>
            <SessionEditor
              session={spec.risk.session}
              onChange={(next) => update((draft) => void (draft.risk.session = next))}
            />
            <Messages messages={errorsFor(errors, "risk")} />
          </Panel>
        </div>

        <div className="builder-side stack">
          <FeedbackPanel
            preview={preview}
            busy={previewBusy}
            stale={previewStale}
            error={previewError}
            valid={valid}
            validating={validating}
            validation={validation}
            onRefresh={() => void runPreview()}
          />

          <Panel title="Period and costs">
            <div className="form-grid">
              <Field label="from (UTC)" htmlFor="preview-start" hint="Inclusive.">
                <input
                  id="preview-start"
                  type="date"
                  value={start}
                  onChange={(event) => setStart(event.target.value)}
                />
              </Field>
              <Field label="to (UTC)" htmlFor="preview-end" hint="Exclusive.">
                <input
                  id="preview-end"
                  type="date"
                  value={end}
                  onChange={(event) => setEnd(event.target.value)}
                />
              </Field>
              <Field label="equity" htmlFor="preview-equity" hint="Account currency.">
                <NumberField
                  id="preview-equity"
                  min={1}
                  step="any"
                  value={equity}
                  onChange={(value) => setEquity(value ?? 1)}
                />
              </Field>
              <Field
                label="spread mode"
                htmlFor="preview-spread-mode"
                hint="Above M1, per bar is rebuilt from the M1 sample."
              >
                <select
                  id="preview-spread-mode"
                  value={spreadMode}
                  onChange={(event) =>
                    setSpreadMode(event.target.value as RunConfigIn["spread_mode"])
                  }
                >
                  {vocabulary.spread_modes.map((mode) => (
                    <option key={mode} value={mode}>
                      {mode}
                    </option>
                  ))}
                </select>
              </Field>
              {spreadMode !== "per_bar" && (
                <Field
                  label={spreadMode === "fixed" ? "spread (points)" : "quantile (0–1)"}
                  htmlFor="preview-spread-value"
                  hint={
                    spreadMode === "fixed"
                      ? "Measured on M1, not read off the bar."
                      : "A level from the observed distribution."
                  }
                >
                  <NumberField
                    id="preview-spread-value"
                    min={0}
                    step="any"
                    value={spreadValue === "" ? null : spreadValue}
                    onChange={(value) => setSpreadValue(value ?? "")}
                  />
                </Field>
              )}
              <Field
                label="commission / lot / side"
                htmlFor="preview-commission"
                hint="Account currency."
              >
                <NumberField
                  id="preview-commission"
                  min={0}
                  step="any"
                  value={commission}
                  onChange={(value) => setCommission(value ?? 0)}
                />
              </Field>
            </div>
          </Panel>

          <Panel
            title="Spec"
            aside={<span className="badge mute">read-only</span>}
            tight
          >
            <pre className="json-view mono" aria-label="the spec as JSON">
              {JSON.stringify(spec, null, 2)}
            </pre>
          </Panel>
        </div>
      </div>
    </div>
  );
}

// -- pieces ----------------------------------------------------------------

function Messages({ messages }: { messages: string[] }) {
  if (!messages.length) return null;
  return (
    <ul className="node-errors" role="alert">
      {messages.map((message) => (
        <li key={message}>{message}</li>
      ))}
    </ul>
  );
}

function uniqueId(base: string, existing: IndicatorEntry[]): string {
  if (!existing.some((entry) => entry.id === base)) return base;
  let index = 2;
  while (existing.some((entry) => entry.id === `${base}${index}`)) index += 1;
  return `${base}${index}`;
}

function IndicatorRow({
  indicator,
  vocabulary,
  errors,
  onChange,
  onRemove,
}: {
  indicator: IndicatorEntry;
  vocabulary: Vocabulary;
  errors: string[];
  onChange: (next: IndicatorEntry) => void;
  onRemove: () => void;
}) {
  const definition = vocabulary.indicators.find((entry) => entry.name === indicator.type);
  return (
    <div className="node leaf">
      <div className="node-head">
        <strong className="node-op">{indicator.id}</strong>
        <span className="node-hint">
          {(definition?.outputs ?? []).length
            ? `outputs: ${(definition?.outputs ?? []).join(", ")}`
            : "single series"}
        </span>
        <div className="node-actions">
          <button type="button" onClick={onRemove}>
            Remove
          </button>
        </div>
      </div>
      <Messages messages={errors} />
      <div className="form-grid">
        <Field label="id" hint="How conditions refer to it.">
          <input
            value={indicator.id}
            onChange={(event) => onChange({ ...indicator, id: event.target.value })}
          />
        </Field>
        <Field label="type" hint="From the registry.">
          <select
            value={indicator.type}
            onChange={(event) => {
              const type = event.target.value;
              const params: Record<string, unknown> = {};
              const next = vocabulary.indicators.find((entry) => entry.name === type);
              for (const param of next?.params ?? []) {
                if (param.default !== null && param.default !== undefined) {
                  params[param.name] = param.default;
                }
              }
              onChange({ ...indicator, type, params });
            }}
          >
            {vocabulary.indicators.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </select>
        </Field>
        {(definition?.params ?? []).map((param) => (
          <Field
            key={param.name}
            label={param.name}
            hint={
              param.choices
                ? "one of the registry's sources"
                : param.minimum !== null && param.minimum !== undefined
                  ? `at least ${param.minimum}`
                  : ""
            }
          >
            {param.choices ? (
              <select
                value={String(indicator.params[param.name] ?? param.default ?? "")}
                onChange={(event) =>
                  onChange({
                    ...indicator,
                    params: { ...indicator.params, [param.name]: event.target.value },
                  })
                }
              >
                {param.choices.map((choice) => (
                  <option key={choice} value={choice}>
                    {choice}
                  </option>
                ))}
              </select>
            ) : (
              <NumberField
                step={param.type === "integer" ? 1 : "any"}
                min={param.minimum ?? undefined}
                value={Number(indicator.params[param.name] ?? param.default ?? 0)}
                onChange={(value) =>
                  onChange({
                    ...indicator,
                    params: { ...indicator.params, [param.name]: value ?? 0 },
                  })
                }
              />
            )}
          </Field>
        ))}
      </div>
    </div>
  );
}

function ConditionSlot({
  label,
  condition,
  path,
  shared,
  onChange,
}: {
  label: string;
  condition: Condition | null;
  path: string;
  shared: Parameters<typeof ConditionEditor>[0]["shared"];
  onChange: (next: Condition | null) => void;
}) {
  return (
    <div className="slot">
      <div className="slot-head">
        <h3>{label}</h3>
        {condition ? (
          <button type="button" onClick={() => onChange(null)}>
            Remove {label.toLowerCase()}
          </button>
        ) : (
          <button type="button" onClick={() => onChange(emptyCondition())}>
            Add {label.toLowerCase()}
          </button>
        )}
      </div>
      {condition ? (
        <ConditionEditor
          condition={condition}
          path={path}
          shared={shared}
          onChange={(next) => onChange(next)}
        />
      ) : (
        <p className="footnote">Not configured.</p>
      )}
    </div>
  );
}

function LevelEditor({
  label,
  level,
  atrIndicators,
  vocabulary,
  errors,
  onChange,
}: {
  label: string;
  level: Level | null;
  atrIndicators: string[];
  vocabulary: Vocabulary;
  errors: string[];
  onChange: (next: Level | null) => void;
}) {
  const type = level?.type ?? "none";
  return (
    <Field
      label={label}
      hint={
        type === "atr"
          ? "A multiple of the ATR read on the signal bar, frozen for the trade."
          : type === "percent"
            ? "A percentage of the fill price."
            : type === "points"
              ? "A fixed distance in instrument points."
              : "Not configured."
      }
    >
      <div className="row">
        <select
          value={type}
          onChange={(event) => {
            const next = event.target.value;
            if (next === "none") return onChange(null);
            if (next === "atr") {
              return onChange({
                type: "atr",
                indicator: atrIndicators[0] ?? "",
                mult: 2,
              });
            }
            onChange({ type: next as "points" | "percent", value: next === "points" ? 150 : 0.5 });
          }}
          aria-label={`${label}: kind`}
        >
          <option value="none">none</option>
          {vocabulary.exit_level_types.map((entry) => (
            <option key={entry} value={entry}>
              {entry}
            </option>
          ))}
        </select>
        {level?.type === "atr" ? (
          <>
            <select
              value={level.indicator}
              onChange={(event) => onChange({ ...level, indicator: event.target.value })}
              aria-label={`${label}: ATR indicator`}
            >
              {atrIndicators.length === 0 && <option value="">add an atr indicator</option>}
              {atrIndicators.map((entry) => (
                <option key={entry} value={entry}>
                  {entry}
                </option>
              ))}
            </select>
            <NumberField
              step="any"
              min={0}
              value={level.mult}
              onChange={(value) => onChange({ ...level, mult: value ?? 0 })}
              aria-label={`${label}: multiple`}
            />
          </>
        ) : level ? (
          <NumberField
            step="any"
            min={0}
            value={level.value}
            onChange={(value) => onChange({ ...level, value: value ?? 0 })}
            aria-label={`${label}: value`}
          />
        ) : null}
      </div>
      <Messages messages={errors} />
    </Field>
  );
}

function SessionEditor({
  session,
  onChange,
}: {
  session: Spec["risk"]["session"];
  onChange: (next: Spec["risk"]["session"]) => void;
}) {
  return (
    <div className="slot">
      <div className="slot-head">
        <h3>Session window</h3>
        {session ? (
          <button type="button" onClick={() => onChange(null)}>
            Remove window
          </button>
        ) : (
          <button
            type="button"
            onClick={() => onChange({ start: "08:00", end: "20:00", timezone: "server" })}
          >
            Add window
          </button>
        )}
      </div>
      {session ? (
        <div className="form-grid">
          <Field label="from" hint="HH:MM">
            <input
              value={session.start}
              onChange={(event) => onChange({ ...session, start: event.target.value })}
            />
          </Field>
          <Field label="to" hint="HH:MM">
            <input
              value={session.end}
              onChange={(event) => onChange({ ...session, end: event.target.value })}
            />
          </Field>
          <Field label="clock" hint="Server time is Europe/Athens on this broker.">
            <select
              value={session.timezone}
              onChange={(event) =>
                onChange({ ...session, timezone: event.target.value as "server" | "utc" })
              }
            >
              <option value="server">server</option>
              <option value="utc">UTC</option>
            </select>
          </Field>
        </div>
      ) : (
        <p className="footnote">Entries allowed at any hour the market is open.</p>
      )}
    </div>
  );
}

// -- the feedback panel ----------------------------------------------------

function FeedbackPanel({
  preview,
  busy,
  stale,
  error,
  valid,
  validating,
  validation,
  onRefresh,
}: {
  preview: Preview | null;
  busy: boolean;
  stale: boolean;
  error: unknown;
  valid: boolean;
  validating: boolean;
  validation: ValidateResponse | null;
  onRefresh: () => void;
}) {
  const attempts = preview?.attempts ?? null;
  const breakeven = preview?.breakeven ?? null;
  const ambiguity = preview?.ambiguity ?? null;
  const tradability = preview?.tradability ?? null;

  return (
    <Panel
      title="Before you run it"
      aside={
        <div className="actions">
          {validating ? (
            <span className="badge info">checking…</span>
          ) : valid ? (
            <span className="badge ok">spec valid</span>
          ) : (
            <span className="badge bad">spec invalid</span>
          )}
          <button type="button" onClick={onRefresh} disabled={!valid || busy}>
            Refresh
          </button>
        </div>
      }
      tight
    >
      {!valid && (
        <div className="panel-body">
          <Notice kind="warn" title="Nothing to preview yet">
            <div>{validation?.message ?? "The spec does not validate."}</div>
            <div style={{ marginTop: 4 }}>
              The messages are shown next to the fields that caused them.
            </div>
          </Notice>
        </div>
      )}

      {valid && !preview && (busy || stale) && (
        <div className="panel-body">
          <Loading label="Counting the signals this spec would produce" height={140} />
        </div>
      )}

      {valid && error ? (
        <div className="panel-body">
          <ErrorNotice error={error} />
        </div>
      ) : null}

      {preview && (
        <>
          {stale && (
            <div className="panel-body">
              <Notice kind="warn" title="These numbers are one edit behind">
                The spec changed after they were computed. Refreshing in a moment.
              </Notice>
            </div>
          )}

          <div className="kpis">
            <div className="kpi">
              <div className="label">Trades at most</div>
              <div className={`value ${preview.judgeable ? "" : "neg"}`}>
                {int(preview.trades_upper_bound)}
              </div>
              <div className="sub">
                {preview.judgeable
                  ? `at least ${preview.min_judgeable_trades} needed — met`
                  : `under the ${preview.min_judgeable_trades} needed to judge anything`}
              </div>
            </div>
            <div className="kpi">
              <div className="label">Break-even win rate</div>
              <div className="value small">
                {breakeven?.valid && breakeven.breakeven_win_rate !== null
                  ? pct(breakeven.breakeven_win_rate, 1)
                  : "—"}
              </div>
              <div className="sub">
                {breakeven?.valid
                  ? `stop ${num(breakeven.loss_points, 0)} pt vs target ${num(
                      breakeven.win_points,
                      0,
                    )} pt`
                  : (breakeven?.reason ?? "not computable")}
              </div>
            </div>
            <div className="kpi">
              <div className="label">Ambiguous bars</div>
              <div
                className={`value small ${
                  ambiguity?.exceeds_threshold ? "neg" : ""
                }`}
              >
                {ambiguity?.expected_ambiguous_share !== null &&
                ambiguity?.expected_ambiguous_share !== undefined
                  ? pct(ambiguity.expected_ambiguous_share, 1)
                  : "—"}
              </div>
              <div className="sub">a priori, upper bound</div>
            </div>
            {/* Two corrections, two thresholds, one observed value. The
                headline is the whole search, because that is the bar a claim
                of discovery has to clear and the one the campaign report
                quotes; the per-instrument figure sits under it, labelled, so
                the two can never be mistaken for each other. */}
            <div className="kpi">
              <div className="label">Sharpe/trade needed</div>
              <div className="value small">
                {attempts?.overall_required_sharpe_per_trade !== null &&
                attempts?.overall_required_sharpe_per_trade !== undefined
                  ? signedMoney(attempts.overall_required_sharpe_per_trade, 4)
                  : "—"}
              </div>
              <div className="sub">
                whole search — {int(attempts?.overall_attempts ?? 0)} attempt
                {(attempts?.overall_attempts ?? 0) === 1 ? "" : "s"}
              </div>
              <div className="sub">
                {attempts?.required_sharpe_per_trade !== null &&
                attempts?.required_sharpe_per_trade !== undefined
                  ? signedMoney(attempts.required_sharpe_per_trade, 4)
                  : "—"}{" "}
                on this instrument — {int(attempts?.attempts ?? 0)} attempt
                {(attempts?.attempts ?? 0) === 1 ? "" : "s"}
              </div>
            </div>
          </div>

          <div className="panel-body stack">
            <Notice kind={preview.judgeable ? "info" : "warn"}>
              {preview.verdict}
            </Notice>

            <dl className="facts">
              <dt>Signals</dt>
              <dd>
                {int(preview.signals_total)} over {int(preview.bars)} bars —{" "}
                {int(preview.signals_long)} long, {int(preview.signals_short)} short,{" "}
                {num(preview.signals_per_1000_bars, 2)} per 1000 bars
              </dd>
              <dt>Spread charged</dt>
              <dd>
                {preview.median_spread_points === null
                  ? "unknown"
                  : `${num(preview.median_spread_points, 1)} points`}{" "}
                <span className="muted">({preview.spread_source})</span>
              </dd>
              <dt>Tradability</dt>
              <dd>
                {tradability ? (
                  <>
                    <Badge
                      kind={
                        !tradability.judged
                          ? "mute"
                          : tradability.tradable
                            ? "ok"
                            : "bad"
                      }
                    >
                      {!tradability.judged
                        ? "not judged"
                        : tradability.tradable
                          ? "testable"
                          : "excluded"}
                    </Badge>{" "}
                    {tradability.reason}
                  </>
                ) : (
                  "not measured"
                )}
              </dd>
              <dt>Attempts</dt>
              <dd>
                {attempts?.verdict ?? "—"}
                {attempts ? (
                  <div className="scope-scroll">
                  <table className="scope-table">
                    <caption className="sr-only">
                      The multiple-testing correction at each scope
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col">Scope</th>
                        <th scope="col">Attempts</th>
                        <th scope="col">Free by luck</th>
                        <th scope="col">Needed at 95%</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th scope="row">
                          Whole search
                          <span className="muted">{attempts.overall_scope}</span>
                        </th>
                        <td className="num">{int(attempts.overall_attempts)}</td>
                        <td className="num">
                          {attempts.overall_expected_max_sharpe != null
                            ? signedMoney(attempts.overall_expected_max_sharpe, 4)
                            : "—"}
                        </td>
                        <td className="num">
                          {attempts.overall_required_sharpe_per_trade != null
                            ? signedMoney(attempts.overall_required_sharpe_per_trade, 4)
                            : "—"}
                        </td>
                      </tr>
                      <tr>
                        <th scope="row">
                          This instrument
                          <span className="muted">{attempts.scope}</span>
                        </th>
                        <td className="num">{int(attempts.attempts)}</td>
                        <td className="num">
                          {attempts.expected_max_sharpe != null
                            ? signedMoney(attempts.expected_max_sharpe, 4)
                            : "—"}
                        </td>
                        <td className="num">
                          {attempts.required_sharpe_per_trade != null
                            ? signedMoney(attempts.required_sharpe_per_trade, 4)
                            : "—"}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                  </div>
                ) : null}
              </dd>
            </dl>

            {(preview.warnings ?? []).length > 0 && (
              <ul className="caveats">
                {(preview.warnings ?? []).map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}

      {valid && !preview && !busy && !stale && !error && (
        <div className="panel-body">
          <Empty title="No preview yet">
            <p className="footnote">
              Pick a period on the right and the numbers appear on their own.
            </p>
          </Empty>
        </div>
      )}
    </Panel>
  );
}
