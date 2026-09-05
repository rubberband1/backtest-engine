"""Measures the dashboard's colour tokens against WCAG, in both themes.

    python -m scripts.check_contrast          table, exit 1 on any failure
    python -m scripts.check_contrast --quiet  failures only

Reads the tokens straight out of `ui/src/styles.css` rather than keeping a
second copy of them, so a token edited in the stylesheet is what gets
measured. Every pair below is one that actually occurs on screen: a
foreground the application paints on that background somewhere. Contrast has
been wrong here before, and it was wrong in the place it usually is - a
muted grey on a light panel, and a lifted green on a dark one.

`tests/test_contrast.py` runs this, so the palette cannot regress quietly.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STYLESHEET = ROOT / "ui" / "src" / "styles.css"

# WCAG 2.2: 4.5:1 for body text, 3:1 for large text and for the parts of a
# control that identify it.
TEXT = 4.5
UI = 3.0

Pair = tuple[str, str, float, str]

# (foreground, background, minimum, where it appears)
PAIRS: tuple[Pair, ...] = (
    ("ink", "bg", TEXT, "body text on a panel"),
    ("ink", "bg-soft", TEXT, "body text on the page ground"),
    ("ink", "bg-sunken", TEXT, "body text on a sunken block"),
    ("ink-soft", "bg", TEXT, "labels, captions, table headers"),
    ("ink-soft", "bg-soft", TEXT, "panel-head labels"),
    ("ink-soft", "bg-sunken", TEXT, "sticky table headers"),
    ("ink-faint", "bg", TEXT, "field hints, sub-labels"),
    ("ink-faint", "bg-soft", TEXT, "hints over the page ground"),
    ("ink-faint", "bg-sunken", TEXT, "column sub-headers"),
    ("pos", "bg", TEXT, "a positive number"),
    ("neg", "bg", TEXT, "a negative number"),
    ("warn", "bg", TEXT, "a flagged value"),
    ("pos", "pos-soft", TEXT, "the PASSED badge"),
    ("neg", "neg-soft", TEXT, "the REFUSED badge"),
    ("warn", "warn-soft", TEXT, "the synthetic-data banner"),
    ("ink", "pos-soft", TEXT, "the body of a success notice"),
    ("ink", "neg-soft", TEXT, "the body of an error notice"),
    ("ink", "warn-soft", TEXT, "the body of a warning notice"),
    ("ink", "accent-soft", TEXT, "the body of an informational notice"),
    ("on-accent", "accent", TEXT, "the primary button, the current tab"),
    ("line-strong", "bg", UI, "the border of an input or a select"),
    ("line-strong", "bg-soft", UI, "an input over the page ground"),
    ("accent", "bg", UI, "the keyboard focus ring"),
    ("accent", "bg-soft", UI, "the focus ring over the page ground"),
    ("ink", "line", UI, "the fill of the progress bar on its track"),
)

# Reported, not enforced. `--line` separates panels and table rows; nothing
# about identifying a control depends on seeing it, and holding a hairline
# divider to 3:1 would make every table look like a spreadsheet grid.
INFORMATIONAL: tuple[tuple[str, str, str], ...] = (
    ("line", "bg", "panel borders and row rules"),
    ("line", "bg-soft", "panel borders over the page ground"),
)


# -- colour --------------------------------------------------------------


def oklch_to_srgb(lightness: float, chroma: float, hue: float) -> tuple[float, float, float]:
    """OKLCH to gamma-encoded sRGB, clipped to the gamut.

    Clipping rather than mapping: every token here is inside sRGB by
    construction, and a token that ever leaves it should be visible as a
    changed number rather than quietly pulled back by a gamut mapper.
    """
    angle = math.radians(hue)
    a = chroma * math.cos(angle)
    b = chroma * math.sin(angle)

    long_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    medium = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    short = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3

    linear = (
        4.0767416621 * long_ - 3.3077115913 * medium + 0.2309699292 * short,
        -1.2684380046 * long_ + 2.6097574011 * medium - 0.3413193965 * short,
        -0.0041960863 * long_ - 0.7034186147 * medium + 1.7076147010 * short,
    )
    return tuple(_encode(min(max(channel, 0.0), 1.0)) for channel in linear)  # type: ignore[return-value]


def _encode(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * channel ** (1 / 2.4) - 0.055


def _decode(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def luminance(srgb: tuple[float, float, float]) -> float:
    red, green, blue = (_decode(channel) for channel in srgb)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def hex_of(srgb: tuple[float, float, float]) -> str:
    red, green, blue = (round(channel * 255) for channel in srgb)
    return f"#{red:02x}{green:02x}{blue:02x}"


# -- the stylesheet ------------------------------------------------------

TOKEN = re.compile(
    r"--([a-z0-9-]+):\s*oklch\(\s*([\d.]+)%\s+([\d.]+)\s+([\d.]+)\s*\)\s*;"
)
BLOCK = re.compile(r"(:root|\[data-theme=\"dark\"\])\s*\{(.*?)\n\}", re.S)


def read_themes(path: Path = STYLESHEET) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Every plain OKLCH token in the light and dark blocks.

    Tokens carrying an alpha (the shadows) are skipped: they sit over an
    unknown background, so there is no pair to measure them in.
    """
    text = path.read_text(encoding="utf-8")
    themes: dict[str, dict[str, tuple[float, float, float]]] = {}
    for selector, body in BLOCK.findall(text):
        name = "dark" if "dark" in selector else "light"
        colours = {
            token: oklch_to_srgb(float(lightness) / 100, float(chroma), float(hue))
            for token, lightness, chroma, hue in TOKEN.findall(body)
        }
        # a theme inherits every token it does not restate
        themes[name] = {**themes.get("light", {}), **colours}
    return themes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="print failures only")
    args = parser.parse_args()

    themes = read_themes()
    missing = set()
    failures: list[str] = []

    for theme in ("light", "dark"):
        tokens = themes.get(theme)
        if not tokens:
            failures.append(f"no {theme} token block in {STYLESHEET.name}")
            continue
        if not args.quiet:
            print(f"\n=== {theme.upper()} ===")
            print(f"{'ratio':>7}  {'':2} {'foreground':<12} on {'background':<12}  where")

        for foreground, background, minimum, where in PAIRS:
            if foreground not in tokens or background not in tokens:
                missing.update({foreground, background} - set(tokens))
                continue
            ratio = contrast(tokens[foreground], tokens[background])
            passed = ratio >= minimum
            if not passed:
                failures.append(
                    f"{theme}: --{foreground} on --{background} is {ratio:.2f}:1, "
                    f"needs {minimum}:1 ({where})"
                )
            if not args.quiet:
                mark = "ok" if passed else "NO"
                print(
                    f"{ratio:>6.2f}:1 {mark:>2} --{foreground:<10} on --{background:<12} {where}"
                )

        if not args.quiet:
            for foreground, background, where in INFORMATIONAL:
                if foreground in tokens and background in tokens:
                    ratio = contrast(tokens[foreground], tokens[background])
                    print(
                        f"{ratio:>6.2f}:1 -- --{foreground:<10} on --{background:<12} "
                        f"{where} (not enforced)"
                    )

    if missing:
        failures.append(f"tokens referenced by a pair but not defined: {sorted(missing)}")

    print()
    if failures:
        for line in failures:
            print(f"FAIL  {line}")
        print(f"\n{len(failures)} contrast failure(s)")
        return 1
    print(f"{len(PAIRS) * 2} pairs checked across two themes, all pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
