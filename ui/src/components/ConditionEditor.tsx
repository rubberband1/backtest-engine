import { NumberField } from "./NumberField";
/**
 * The condition tree, edited with controls instead of typed as JSON.
 *
 * One recursive component. A node is either a group (`and` / `or`), a
 * negation, or a leaf that compares operands; the editor renders whichever
 * it is and offers the moves that make sense there - change the operator,
 * swap an operand, wrap the node in a group, delete it.
 *
 * Two decisions worth stating:
 *
 * - **Nothing is invented on a type change.** Turning a comparison into a
 *   `between` keeps the operand already chosen and adds the two bounds; the
 *   reverse drops them. A change that quietly rewrites what was built is
 *   worse than one that refuses.
 * - **Errors are shown where they are caused.** The API validates the whole
 *   spec and answers with paths (`entry.long.operands.0.left`), so each node
 *   knows which messages are its own and prints them under itself rather
 *   than in a list at the bottom of the page.
 */
import type { Vocabulary } from "../api/client";
import {
  COMPARE_LABELS,
  type Condition,
  type Operand,
  type OperandKind,
  OPERAND_KINDS,
  clone,
  emptyCondition,
  emptyOperand,
  errorsFor,
  isBetween,
  isCompare,
  isGroup,
  isNot,
  isTrend,
  operandKind,
} from "../strategy/spec";

interface Shared {
  vocabulary: Vocabulary;
  indicatorRefs: string[];
  errors: Map<string, string[]>;
}

function FieldErrors({ messages }: { messages: string[] }) {
  if (!messages.length) return null;
  return (
    <ul className="node-errors" role="alert">
      {messages.map((message) => (
        <li key={message}>{message}</li>
      ))}
    </ul>
  );
}

function OperandEditor({
  operand,
  path,
  label,
  shared,
  onChange,
}: {
  operand: Operand;
  path: string;
  label: string;
  shared: Shared;
  onChange: (next: Operand) => void;
}) {
  const kind = operandKind(operand);
  const messages = errorsFor(shared.errors, path);
  const selectId = `${path}-kind`;
  const valueId = `${path}-value`;

  function changeKind(next: OperandKind) {
    const first =
      next === "ref"
        ? shared.indicatorRefs[0]
        : next === "feature"
          ? shared.vocabulary.features[0]
          : next === "bar"
            ? "close"
            : undefined;
    onChange(emptyOperand(next, first));
  }

  const options =
    kind === "ref"
      ? shared.indicatorRefs
      : kind === "bar"
        ? shared.vocabulary.bar_fields
        : kind === "feature"
          ? shared.vocabulary.features
          : [];

  return (
    <div className="operand">
      <label className="operand-label" htmlFor={selectId}>
        {label}
      </label>
      <div className="operand-controls">
        <select
          id={selectId}
          value={kind}
          onChange={(event) => changeKind(event.target.value as OperandKind)}
          aria-label={`${label}: kind of value`}
        >
          {OPERAND_KINDS.map((entry) => (
            <option key={entry.kind} value={entry.kind}>
              {entry.label}
            </option>
          ))}
        </select>
        {kind === "const" ? (
          <NumberField
            id={valueId}
            step="any"
            value={"const" in operand ? operand.const : 0}
            onChange={(value) => onChange({ const: value ?? 0 })}
            aria-label={`${label}: constant`}
          />
        ) : (
          <select
            id={valueId}
            value={
              "ref" in operand
                ? operand.ref
                : "bar" in operand
                  ? operand.bar
                  : "feature" in operand
                    ? operand.feature
                    : ""
            }
            onChange={(event) => {
              const value = event.target.value;
              onChange(
                kind === "ref"
                  ? { ref: value }
                  : kind === "bar"
                    ? { bar: value }
                    : { feature: value },
              );
            }}
            aria-label={`${label}: value`}
          >
            {options.length === 0 && <option value="">no option available</option>}
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        )}
      </div>
      <FieldErrors messages={messages} />
    </div>
  );
}

const LEAF_OPS = [
  "gt",
  "gte",
  "lt",
  "lte",
  "eq",
  "cross_above",
  "cross_below",
  "between",
  "rising",
  "falling",
] as const;

export function ConditionEditor({
  condition,
  path,
  shared,
  depth = 0,
  onChange,
  onRemove,
}: {
  condition: Condition;
  path: string;
  shared: Shared;
  depth?: number;
  onChange: (next: Condition) => void;
  onRemove?: () => void;
}) {
  const messages = errorsFor(shared.errors, path);

  function wrap(op: "and" | "or") {
    onChange({ op, operands: [clone(condition), emptyCondition()] });
  }

  function changeLeafOp(next: string) {
    const current = clone(condition);
    if (next === "between") {
      const left = isCompare(current) || isBetween(current) ? current.left : { const: 0 };
      onChange({ op: "between", left, low: { const: 0 }, high: { const: 100 } });
      return;
    }
    if (next === "rising" || next === "falling") {
      const operand = isTrend(current)
        ? current.operand
        : isCompare(current) || isBetween(current)
          ? current.left
          : { const: 0 };
      onChange({ op: next, operand, periods: isTrend(current) ? current.periods : 1 });
      return;
    }
    const left = isTrend(current)
      ? current.operand
      : isCompare(current) || isBetween(current)
        ? current.left
        : { const: 0 };
    const right = isCompare(current) ? current.right : { const: 0 };
    onChange({ op: next as never, left, right });
  }

  // -- groups -------------------------------------------------------------

  if (isGroup(condition)) {
    return (
      <div className={`node group depth-${Math.min(depth, 3)}`}>
        <div className="node-head">
          <select
            value={condition.op}
            onChange={(event) =>
              onChange({ ...condition, op: event.target.value as "and" | "or" })
            }
            aria-label="group operator"
          >
            <option value="and">all of (AND)</option>
            <option value="or">any of (OR)</option>
          </select>
          <span className="node-hint">
            {condition.operands.length} condition
            {condition.operands.length === 1 ? "" : "s"}
          </span>
          <div className="node-actions">
            <button
              type="button"
              onClick={() =>
                onChange({ ...condition, operands: [...condition.operands, emptyCondition()] })
              }
            >
              Add condition
            </button>
            <button
              type="button"
              onClick={() => onChange({ op: "not", operand: clone(condition) })}
            >
              Negate
            </button>
            {onRemove && (
              <button type="button" onClick={onRemove}>
                Remove
              </button>
            )}
          </div>
        </div>
        <FieldErrors messages={messages} />
        <div className="node-children">
          {condition.operands.map((child, index) => (
            <ConditionEditor
              key={index}
              condition={child}
              path={`${path}.operands.${index}`}
              shared={shared}
              depth={depth + 1}
              onChange={(next) => {
                const operands = [...condition.operands];
                operands[index] = next;
                onChange({ ...condition, operands });
              }}
              onRemove={
                condition.operands.length > 1
                  ? () =>
                      onChange({
                        ...condition,
                        operands: condition.operands.filter((_, i) => i !== index),
                      })
                  : undefined
              }
            />
          ))}
        </div>
      </div>
    );
  }

  // -- negation -----------------------------------------------------------

  if (isNot(condition)) {
    return (
      <div className={`node negation depth-${Math.min(depth, 3)}`}>
        <div className="node-head">
          <strong className="node-op">NOT</strong>
          <div className="node-actions">
            <button type="button" onClick={() => onChange(clone(condition.operand))}>
              Un-negate
            </button>
            {onRemove && (
              <button type="button" onClick={onRemove}>
                Remove
              </button>
            )}
          </div>
        </div>
        <FieldErrors messages={messages} />
        <div className="node-children">
          <ConditionEditor
            condition={condition.operand}
            path={`${path}.operand`}
            shared={shared}
            depth={depth + 1}
            onChange={(next) => onChange({ op: "not", operand: next })}
          />
        </div>
      </div>
    );
  }

  // -- leaves -------------------------------------------------------------

  return (
    <div className={`node leaf depth-${Math.min(depth, 3)}`}>
      <div className="node-head">
        <select
          value={condition.op}
          onChange={(event) => changeLeafOp(event.target.value)}
          aria-label="condition"
        >
          {LEAF_OPS.map((op) => (
            <option key={op} value={op}>
              {COMPARE_LABELS[op] ?? op}
            </option>
          ))}
        </select>
        <div className="node-actions">
          <button type="button" onClick={() => wrap("and")}>
            Group with AND
          </button>
          <button type="button" onClick={() => wrap("or")}>
            Group with OR
          </button>
          <button type="button" onClick={() => onChange({ op: "not", operand: clone(condition) })}>
            Negate
          </button>
          {onRemove && (
            <button type="button" onClick={onRemove}>
              Remove
            </button>
          )}
        </div>
      </div>
      <FieldErrors messages={messages} />

      <div className="operands">
        {isTrend(condition) ? (
          <>
            <OperandEditor
              operand={condition.operand}
              path={`${path}.operand`}
              label="value"
              shared={shared}
              onChange={(next) => onChange({ ...condition, operand: next })}
            />
            <div className="operand">
              <label className="operand-label" htmlFor={`${path}-periods`}>
                bars
              </label>
              <div className="operand-controls">
                <NumberField
                  id={`${path}-periods`}
                  min={1}
                  step={1}
                  value={condition.periods}
                  onChange={(value) =>
                    onChange({ ...condition, periods: value ?? 1 })
                  }
                />
              </div>
              <FieldErrors messages={errorsFor(shared.errors, `${path}.periods`)} />
            </div>
          </>
        ) : isBetween(condition) ? (
          <>
            <OperandEditor
              operand={condition.left}
              path={`${path}.left`}
              label="value"
              shared={shared}
              onChange={(next) => onChange({ ...condition, left: next })}
            />
            <OperandEditor
              operand={condition.low}
              path={`${path}.low`}
              label="lower bound"
              shared={shared}
              onChange={(next) => onChange({ ...condition, low: next })}
            />
            <OperandEditor
              operand={condition.high}
              path={`${path}.high`}
              label="upper bound"
              shared={shared}
              onChange={(next) => onChange({ ...condition, high: next })}
            />
          </>
        ) : (
          <>
            <OperandEditor
              operand={condition.left}
              path={`${path}.left`}
              label="left"
              shared={shared}
              onChange={(next) => onChange({ ...condition, left: next })}
            />
            <OperandEditor
              operand={condition.right}
              path={`${path}.right`}
              label="right"
              shared={shared}
              onChange={(next) => onChange({ ...condition, right: next })}
            />
          </>
        )}
      </div>
    </div>
  );
}

export type { Shared as ConditionEditorShared };
