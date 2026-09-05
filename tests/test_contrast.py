"""The palette, measured rather than looked at.

Contrast has been wrong in this dashboard before, and it was wrong in the two
places it usually is: a muted grey on a light panel, and the border of a form
control against its own background. Both looked fine. Only the arithmetic
said otherwise, which is why the arithmetic runs in the suite.

`scripts/check_contrast.py` reads the tokens out of the stylesheet, so this
measures what the application actually paints - there is no second copy of
the palette to drift from the first.
"""
from __future__ import annotations

import pytest

from scripts.check_contrast import (
    PAIRS,
    contrast,
    luminance,
    oklch_to_srgb,
    read_themes,
)


@pytest.fixture(scope="module")
def themes():
    return read_themes()


def test_both_themes_are_defined(themes):
    assert set(themes) == {"light", "dark"}


def test_every_declared_pair_meets_its_minimum(themes):
    """The whole point. A failure here names the token and the ratio."""
    failures = []
    for theme, tokens in themes.items():
        for foreground, background, minimum, where in PAIRS:
            assert foreground in tokens, f"{theme}: --{foreground} is not defined"
            assert background in tokens, f"{theme}: --{background} is not defined"
            ratio = contrast(tokens[foreground], tokens[background])
            if ratio < minimum:
                failures.append(
                    f"{theme}: --{foreground} on --{background} is {ratio:.2f}:1, "
                    f"needs {minimum}:1 ({where})"
                )
    assert not failures, "\n".join(failures)


def test_the_dark_theme_is_dark_and_the_light_one_is_light(themes):
    """A theme swapped by accident would still pass every contrast pair."""
    assert luminance(themes["light"]["bg"]) > 0.8
    assert luminance(themes["dark"]["bg"]) < 0.1
    assert luminance(themes["light"]["ink"]) < 0.1
    assert luminance(themes["dark"]["ink"]) > 0.8


def test_neutrals_are_tinted_rather_than_dead_grey(themes):
    """Zero chroma reads as a screenshot of a screenshot. Small, but not zero."""
    for theme, tokens in themes.items():
        for name in ("bg", "bg-soft", "ink", "ink-soft"):
            red, green, blue = tokens[name]
            spread = max(red, green, blue) - min(red, green, blue)
            assert spread > 0.0, f"{theme}: --{name} has no tint at all"


def test_the_semantic_colours_stay_apart_in_greyscale(themes):
    """Roughly eight percent of men cannot separate the green from the red.

    Colour is never this application's only carrier - there is always a sign,
    a word or a badge - but the three should still not land on the same
    lightness, or a screenshot in greyscale becomes unreadable too.
    """
    for theme, tokens in themes.items():
        values = {name: luminance(tokens[name]) for name in ("pos", "neg", "warn")}
        pairs = [("pos", "neg"), ("pos", "warn"), ("neg", "warn")]
        for first, second in pairs:
            gap = abs(values[first] - values[second])
            assert gap > 0.02, (
                f"{theme}: --{first} and --{second} differ by {gap:.4f} in "
                f"luminance, which is invisible without colour"
            )


@pytest.mark.parametrize(
    ("lightness", "chroma", "hue", "expected"),
    [
        (1.0, 0.0, 0.0, (1.0, 1.0, 1.0)),
        (0.0, 0.0, 0.0, (0.0, 0.0, 0.0)),
    ],
)
def test_the_colour_conversion_agrees_at_the_ends(lightness, chroma, hue, expected):
    converted = oklch_to_srgb(lightness, chroma, hue)
    for channel, target in zip(converted, expected):
        assert channel == pytest.approx(target, abs=0.002)


def test_contrast_matches_the_wcag_reference_pair():
    """Black on white is 21:1 by definition; anything else means a bug here."""
    black = oklch_to_srgb(0.0, 0.0, 0.0)
    white = oklch_to_srgb(1.0, 0.0, 0.0)
    assert contrast(black, white) == pytest.approx(21.0, abs=0.05)
