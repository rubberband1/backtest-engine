# The interface

What the dashboard looks like, and the rules it follows. The tokens
themselves live in `ui/src/styles.css` and are not repeated here — this is
the part that would otherwise be tacit and drift.

## Register

An instrument, not a landing page. Somebody reading it is in the middle of a
task: comparing three hundred screened cells, deciding whether a number is
worth anything. Density is high, familiarity is a feature, and the interface
is supposed to disappear into the work.

## Light by default, dark by choice

Light is the default and the OS preference is deliberately not consulted.
Someone whose whole system is dark still gets the light theme on first open,
and one click changes it permanently (`localStorage`, applied before the
first paint by an inline script in `index.html`).

The two themes are for two different situations, which is why dark is not the
light theme inverted:

- **Light** is the working state: a lit desk, small type, a lot of numbers.
- **Dark** exists mostly for recording the screen, where a white page blows
  out the frame. It has to stay legible at large type in a vertical crop, so
  depth comes from lighter surfaces rather than shadow, the greens and reds
  are lifted and desaturated to survive a dark ground, and body text drops one
  weight step because light on dark reads heavier than the reverse.

## Colour means something or it is not used

Three colours carry meaning and nothing else does:

| Token    | Meaning   |
|----------|-----------|
| `--pos`  | passed, gained, above the line |
| `--warn` | needs a look before it is trusted |
| `--neg`  | refused, lost, below the line |

There is no brand colour, and no accent. The primary action, the current tab
and the focus ring are **ink** — which is what leaves those three as the only
coloured things on a screen full of figures. The one place colour is spent on
identity rather than meaning is the comparison chart, where six overlaid
equity curves need six distinguishable strokes; that chart carries a labelled
legend, so the colour is never the only key.

**Colour is never the only carrier.** A number's sign is printed, a verdict is
a word, a state is a badge. The three semantic colours are also held apart in
*lightness*, not only in hue, so a greyscale screenshot and a reader who
cannot separate red from green both keep the distinction.
`tests/test_contrast.py` enforces that, along with every foreground-on-
background pair the interface paints:

```
python -m scripts.check_contrast
```

It reads the tokens out of the stylesheet, so there is no second copy of the
palette to drift. It has caught two real defects: input borders at 2.1:1
against their own background, and a green and a red one hundredth of a
luminance apart.

## Type

Two families, both variable, both served from `ui/src/fonts/` — nothing is
fetched at runtime, because the application has to work with no network and
the packaged build has no server to ask.

- **Inter** for text.
- **JetBrains Mono** for every number that changes or lines up in a column,
  with `font-variant-numeric: tabular-nums`. Without both, a column of figures
  dances from row to row.

## Motion

The rule: **movement has to mean something.** Menus, fields, tables and
navigation are instant. Four things move, and nothing else does.

1. **Waiting.** Concentric arcs turning against each other, in CSS
   `transform` only, so the browser runs it on the compositor with no
   JavaScript per frame. Under it the text says *which* work and how far in —
   "screening cell 47 of 300", "permutation 340 / 1000" — never "loading". A
   job whose backend cannot count its steps gets a sweeping bar that means
   "running", not an invented percentage.
2. **A verdict arriving.** The gate-zero outcome, the admissibility verdict,
   the attempts panel and a run's headline metrics enter over 240 ms. Those
   are the moments the application says the thing it exists to say. Nothing
   else gets the treatment.
3. **Curves.** Equity and drawdown draw from the left in 500 ms, once, the
   first time that run's curve appears. Never on a resize, a hover or a
   filter change: a chart that re-animates while you are reading it is a chart
   you cannot read.
4. **The live runner changing state.** Stopped, listening, in position — the
   indicator transitions rather than swapping.

Nothing runs longer than 500 ms, and only the curve reaches it. No parallax,
no scroll reveals, no decorative effects.

`prefers-reduced-motion` makes everything instant **except the waiting
indicators**, which are slowed instead of stopped: freezing those leaves a
reader unable to tell a working screen from a hung one, which is worse than
the motion it spares them.

## Two things that must be visible

- **The live runner**, on every page. A strip under the header that renders
  whether or not it has anything to report — one that appears only when
  something is live is one nobody learns to look at. It reads diaries through
  the API, so opening a page cannot disturb a runner that is trading.
- **Getting the data**, as a command. The Run page has a panel that downloads
  the part of a period the cache does not hold, with the interval it is
  fetching and what arrived. Before it, the only sign that an instrument could
  be extended was that its coverage silently grew after somebody ran a script.

## Deliberate exceptions

- **The close confirmation is a native dialog.** In-page modals are avoided
  everywhere else, but a web page cannot reliably block an operating system
  window from closing, and that particular question — a live runner is still
  trading — is the one that must not be missed.
- **The condition tree keeps a coloured left border.** Elsewhere a heavy side
  stripe is decoration; there it encodes the node's kind and its nesting, and
  it is what keeps a fourth level readable without indenting off a 375 px
  screen.
