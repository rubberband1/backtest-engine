"""Draws the Falsify mark and writes every file that needs a copy of it.

    python -m scripts.make_icons

One geometry, defined once below, rendered to:

    brand/falsify-mark.svg    the mark alone, `currentColor`, for the UI
    brand/falsify-icon.svg    the app icon: mark knocked out of an ink tile
    brand/falsify.ico         the same tile at seven sizes, for Windows
    ui/public/favicon.svg     the mark, following the browser's theme

The mark is an F whose crossbar overshoots the stem and leans thirty degrees,
so the letter reads as struck through: the thing the engine does to a
strategy.

The rasteriser is deliberately not a general SVG renderer - it draws these
three quadrilaterals and nothing else. That is what lets this file run on a
bare Python: no Pillow, no cairo, no headless browser in the build.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRAND = ROOT / "brand"
FAVICON = ROOT / "ui" / "public" / "favicon.svg"

# -- the mark ------------------------------------------------------------

# All coordinates are on a 32x32 grid, the size the mark is drawn for. The
# stem and the arm are axis-aligned; the strike is the same rectangle turned
# about its own centre.
STEM = (10.0, 3.0, 4.5, 26.0)
ARM = (10.0, 3.0, 17.0, 4.5)
STRIKE = (2.5, 16.75, 22.0, 4.5)
STRIKE_PIVOT = (13.5, 19.0)
STRIKE_ANGLE = -30.0  # SVG's y grows downward, so a negative angle leans left

# Ink and paper, matching the two --ink / --bg tokens the dashboard uses. The
# icon carries them literally rather than by reference: a .ico has no
# stylesheet, and a Windows taskbar is as often dark as light, so the mark
# needs its own ground instead of inheriting one.
INK = (0x18, 0x1C, 0x22)
PAPER = (0xF7, 0xF8, 0xFA)
TILE_RADIUS = 7.0
# Share of the tile the mark occupies. Below about 0.65 it looks lost inside
# the tile; above 0.75 the strike's tips crowd the corner rounding at 16px.
MARK_SCALE = 0.70

Quad = tuple[tuple[float, float], ...]


def _corners(rect: tuple[float, float, float, float]) -> Quad:
    x, y, w, h = rect
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def _rotate(quad: Quad, degrees: float, pivot: tuple[float, float]) -> Quad:
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    px, py = pivot
    return tuple(
        (px + (x - px) * cos - (y - py) * sin, py + (x - px) * sin + (y - py) * cos)
        for x, y in quad
    )


def mark_shapes() -> list[Quad]:
    """The three quadrilaterals the mark is made of, on the 32x32 grid."""
    return [
        _corners(STEM),
        _corners(ARM),
        _rotate(_corners(STRIKE), STRIKE_ANGLE, STRIKE_PIVOT),
    ]


def mark_bounds() -> tuple[float, float, float, float]:
    """left, top, right, bottom of the drawn mark.

    Not the 32-grid: the strike leans out past it on the left, so centring
    the mark on the grid puts it visibly off-centre in a tile.
    """
    points = [point for quad in mark_shapes() for point in quad]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return min(xs), min(ys), max(xs), max(ys)


def _inside(quad: Quad, x: float, y: float) -> bool:
    """Point in convex quadrilateral, by consistent side of every edge."""
    sign = 0
    for index in range(len(quad)):
        ax, ay = quad[index]
        bx, by = quad[(index + 1) % len(quad)]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if cross > 0:
            current = 1
        elif cross < 0:
            current = -1
        else:
            continue
        if sign and current != sign:
            return False
        sign = current
    return True


def _in_tile(x: float, y: float, size: float, radius: float) -> bool:
    """Point inside the rounded square that backs the icon."""
    if not (0 <= x <= size and 0 <= y <= size):
        return False
    cx = min(max(x, radius), size - radius)
    cy = min(max(y, radius), size - radius)
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius * radius


# -- rasteriser ----------------------------------------------------------

SAMPLES = 4  # per axis, so sixteen samples a pixel


def render_tile(size: int) -> bytes:
    """RGBA rows for one icon: the mark knocked out of a rounded ink tile.

    Sampled rather than scan-converted. At sixteen samples a pixel the edges
    of a shape this simple are indistinguishable from an analytic coverage
    computation, and the whole set of seven sizes renders in well under a
    second.
    """
    radius = TILE_RADIUS * size / 32.0
    left, top, right, bottom = mark_bounds()
    scale = size * MARK_SCALE / max(right - left, bottom - top)
    offset_x = size / 2 - (left + right) / 2 * scale
    offset_y = size / 2 - (top + bottom) / 2 * scale
    placed = [
        tuple((x * scale + offset_x, y * scale + offset_y) for x, y in quad)
        for quad in mark_shapes()
    ]

    step = 1.0 / SAMPLES
    total = SAMPLES * SAMPLES
    rows = bytearray()
    for row in range(size):
        rows.append(0)  # PNG filter: none
        for column in range(size):
            covered = 0
            marked = 0
            for sy in range(SAMPLES):
                y = row + (sy + 0.5) * step
                for sx in range(SAMPLES):
                    x = column + (sx + 0.5) * step
                    if not _in_tile(x, y, size, radius):
                        continue
                    covered += 1
                    if any(_inside(quad, x, y) for quad in placed):
                        marked += 1
            if not covered:
                rows.extend(b"\x00\x00\x00\x00")
                continue
            # the mark's colour is mixed into the tile's before the tile's own
            # coverage becomes alpha: compositing them separately leaves a
            # dark fringe around the glyph on a light background
            share = marked / covered
            rows.extend(
                bytes(
                    round(ink + (paper - ink) * share)
                    for ink, paper in zip(INK, PAPER)
                )
            )
            rows.append(round(255 * covered / total))
    return bytes(rows)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))


def encode_png(size: int, rows: bytes) -> bytes:
    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows, 9))
        + _chunk(b"IEND", b"")
    )


ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def encode_ico(images: list[tuple[int, bytes]]) -> bytes:
    """A PNG-per-entry .ico, which Windows has read since Vista."""
    directory = struct.pack("<HHH", 0, 1, len(images))
    offset = len(directory) + 16 * len(images)
    entries = bytearray()
    payload = bytearray()
    for size, png in images:
        # 256 is written as 0: the field is one byte
        entries.extend(
            struct.pack(
                "<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(png), offset
            )
        )
        payload.extend(png)
        offset += len(png)
    return directory + bytes(entries) + bytes(payload)


# -- svg -----------------------------------------------------------------


def _svg_shapes(fill: str) -> str:
    """The three rectangles. An empty fill leaves the colour to a stylesheet."""
    x, y, w, h = STRIKE
    px, py = STRIKE_PIVOT
    group = f'<g fill="{fill}">' if fill else "<g>"
    return (
        group + f'<rect x="{STEM[0]:g}" y="{STEM[1]:g}" width="{STEM[2]:g}" height="{STEM[3]:g}"/>'
        f'<rect x="{ARM[0]:g}" y="{ARM[1]:g}" width="{ARM[2]:g}" height="{ARM[3]:g}"/>'
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
        f'transform="rotate({STRIKE_ANGLE:g} {px:g} {py:g})"/>'
        f"</g>"
    )


def _hex(colour: tuple[int, int, int]) -> str:
    red, green, blue = colour
    return f"#{red:02x}{green:02x}{blue:02x}"


def mark_svg() -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" '
        'width="32" height="32" role="img" aria-label="Falsify">'
        + _svg_shapes("currentColor")
        + "</svg>\n"
    )


def icon_svg() -> str:
    left, top, right, bottom = mark_bounds()
    scale = 32 * MARK_SCALE / max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" '
        'width="32" height="32" role="img" aria-label="Falsify">'
        f'<rect width="32" height="32" rx="{TILE_RADIUS:g}" fill="{_hex(INK)}"/>'
        f'<g transform="translate(16 16) scale({scale:.4f}) '
        f'translate({-cx:.3f} {-cy:.3f})">'
        + _svg_shapes(_hex(PAPER))
        + "</g></svg>\n"
    )


def favicon_svg() -> str:
    """The tab icon, which follows the browser's theme rather than the tile.

    A tab strip is the one place the mark sits directly on the browser's own
    chrome, and that chrome is dark or light by the reader's choice - so this
    is the single copy that has no ground of its own.
    """
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" '
        'role="img" aria-label="Falsify">'
        "<style>"
        f"path,rect{{fill:{_hex(INK)}}}"
        f"@media (prefers-color-scheme:dark){{path,rect{{fill:{_hex(PAPER)}}}}}"
        "</style>" + _svg_shapes("") + "</svg>\n"
    )


def main() -> int:
    BRAND.mkdir(parents=True, exist_ok=True)
    FAVICON.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for path, text in (
        (BRAND / "falsify-mark.svg", mark_svg()),
        (BRAND / "falsify-icon.svg", icon_svg()),
        (FAVICON, favicon_svg()),
    ):
        path.write_text(text, encoding="utf-8")
        written.append(path)

    images = [(size, encode_png(size, render_tile(size))) for size in ICO_SIZES]
    ico = BRAND / "falsify.ico"
    ico.write_bytes(encode_ico(images))
    written.append(ico)

    # the largest PNG on its own: PyInstaller wants an .ico, everything else
    # that shows an application icon wants a PNG
    png = BRAND / "falsify-256.png"
    png.write_bytes(dict(images)[256])
    written.append(png)

    for path in written:
        print(f"{path.relative_to(ROOT).as_posix()}  {path.stat().st_size:>7,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
