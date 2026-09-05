/**
 * What the application shows while a job is running.
 *
 * Two pieces, and the second is the important one. The spinner says work is
 * happening; the text says *which* work, and how far in. "Loading" is not an
 * acceptable thing to tell someone who started a thousand-iteration
 * permutation test, and a campaign that reports "screening cell 47 of 300"
 * is the difference between waiting and wondering whether it hung.
 */
import type { ReactNode } from "react";

/**
 * Concentric arcs turning against each other.
 *
 * Only `transform` is animated, entirely in CSS, so the browser runs it on
 * the compositor: no JavaScript per frame and no layout work. The inner ring
 * is dropped below 26px, where three arcs stop being three arcs and become a
 * smudge.
 */
export function Spinner({ size = 18, label }: { size?: number; label?: string }) {
  const rings: { radius: number; arc: number; className: string }[] = [
    { radius: 15, arc: 0.26, className: "ring-outer" },
    { radius: 10.5, arc: 0.3, className: "ring-middle" },
  ];
  if (size >= 26) rings.push({ radius: 6, arc: 0.34, className: "ring-inner" });

  return (
    <svg
      className="spinner"
      width={size}
      height={size}
      viewBox="0 0 40 40"
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      {rings.map((ring) => {
        const circumference = 2 * Math.PI * ring.radius;
        return (
          <circle
            key={ring.className}
            className={ring.className}
            cx="20"
            cy="20"
            r={ring.radius}
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray={`${(circumference * ring.arc).toFixed(2)} ${circumference.toFixed(2)}`}
          />
        );
      })}
    </svg>
  );
}

/**
 * The waiting state of one long job.
 *
 * `completed` and `total` are optional: a job whose backend cannot say how
 * many steps it has gets a sweeping bar that means "running", never an
 * invented percentage. The heights are reserved, so the panel does not grow
 * when the counter first arrives.
 */
export function Progress(props: {
  label: string;
  detail?: ReactNode;
  completed?: number | null;
  total?: number | null;
}) {
  const total = props.total ?? 0;
  const completed = props.completed ?? 0;
  const measured = total > 0;
  const share = measured ? Math.min(1, Math.max(0, completed / total)) : 0;

  return (
    <div className="progress">
      <Spinner size={30} />
      <div className="progress-body">
        <div className="progress-label">
          {props.label}
          {measured && (
            <span className="progress-count">
              {" "}
              {completed.toLocaleString("en-US")} / {total.toLocaleString("en-US")}
            </span>
          )}
        </div>
        <div className="progress-detail">{props.detail}</div>
        <div
          className={`progress-track${measured ? "" : " unmeasured"}`}
          role="progressbar"
          aria-label={props.label}
          aria-valuenow={measured ? completed : undefined}
          aria-valuemin={measured ? 0 : undefined}
          aria-valuemax={measured ? total : undefined}
        >
          <i style={measured ? { width: `${(share * 100).toFixed(1)}%` } : undefined} />
        </div>
      </div>
    </div>
  );
}
