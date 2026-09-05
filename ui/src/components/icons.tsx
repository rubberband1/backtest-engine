/**
 * The icon set. One grid, one stroke weight, one cap style.
 *
 * Sixteen units, 1.5 stroke, round caps and joins, no fill. Drawn here
 * rather than pulled from a library because the application ships six of
 * them and a dependency would be six icons plus a thousand it never renders.
 *
 * Every icon is decorative: each is paired with a word in the interface, so
 * they are hidden from assistive technology and the accessible name comes
 * from the control around them.
 */
import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function Icon({ size = 16, children, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  );
}

export function SunIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <circle cx="8" cy="8" r="3.1" />
      <path d="M8 1.4v1.6M8 13v1.6M2.1 8H3.7M12.3 8h1.6M3.8 3.8l1.2 1.2M11 11l1.2 1.2M12.2 3.8L11 5M5 11l-1.2 1.2" />
    </Icon>
  );
}

export function MoonIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M13.4 9.7A5.9 5.9 0 0 1 6.3 2.6a5.9 5.9 0 1 0 7.1 7.1Z" />
    </Icon>
  );
}

/** Bringing bars down from the broker into the local cache. */
export function DownloadIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 2v7.4M4.9 6.6 8 9.7l3.1-3.1M2.6 12.2v1.2h10.8v-1.2" />
    </Icon>
  );
}

export function CheckIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="m3 8.4 3.2 3.2L13 4.8" />
    </Icon>
  );
}

export function CrossIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Icon>
  );
}

/** Not "something went wrong": something needs a look before it is trusted. */
export function AlertIcon(props: IconProps) {
  return (
    <Icon {...props}>
      <path d="M8 2.6 14.4 13H1.6L8 2.6ZM8 6.6v3.1" />
      <path d="M8 11.6h.01" />
    </Icon>
  );
}
