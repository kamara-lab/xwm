"""Plot palettes, with each one's categorical limit enforced in code.

Three named palettes, chosen by what the chart is doing:

``"blue-orange"`` (the default for **curves** -- losses, training histories,
error-vs-horizon) is the Wong colourblind-safe family. Line charts usually carry
two to four series and need maximum separation between them, and this palette
delivers it: adjacent-pair ΔE of 24-36 in OKLab, passing every check in both
light and dark mode.

``"viridis"`` (the default for **magnitude** -- images, PCA colourbars, and
categorical bars where five configurations must be told apart) is perceptually
uniform, which is what a continuous quantity wants.

``"brand"`` is the two-colour accent pair, :data:`ACCENT` and :data:`AMBER`, for
the very common chart that compares exactly two things -- real against imagined,
before against after -- and should look like it belongs to the rest of the site.
Both are read off the kamara gradient, and they are each other's channels
reversed, so they sit at one lightness. Measured the same way as everything else
below: normal-vision ΔE 27.1, CVD 22.4, inside the band Wong's own adjacent pairs
occupy. It stops at two, because a third hue would have to come from outside the
brand.

Using viridis for *categorical* series means sampling discrete steps from a
sequential ramp. That works, but only up to a point, and the point is measurable
rather than a matter of taste. Sampling the ramp over ``VIRIDIS_SPAN`` and
checking adjacent-pair separation in OKLab gives:

======  ==================  ====================
slots   CVD separation ΔE   normal-vision ΔE
======  ==================  ====================
2       62.8                63.7
3       28.3                33.0
4       18.6                22.4
5       13.6                16.3
6       10.4                **13.2 -- too low**
======  ==================  ====================

A normal-vision ΔE below 15 means readers with full colour vision cannot
reliably tell the pair apart, and no amount of secondary encoding fixes that.
So :func:`viridis_colors` refuses more than :data:`MAX_CATEGORICAL` series and
tells you to use small multiples instead -- a limit that is much more useful
enforced than documented.

Two further consequences of using a sequential ramp categorically:

* Its ends are very dark and very light, so the extreme steps have low contrast
  against the plot surface. Every figure here therefore carries a legend, and
  every example also writes the same numbers out as a table
  (:func:`xwm.plots.save_table`), so nothing depends on reading a colour.
* Its middle is desaturated and reads grey-ish. Series are given distinct
  markers and line styles as well as colours, so identity never rests on hue
  alone.

Chrome
------

Everything that is *not* data -- background, text, gridlines, spines, tick marks
-- comes from the kamara design tokens (https://kamara.dev), so a figure sits on
the same ground as the page around it instead of announcing itself as a white
rectangle:

============  =========  =================================================
token         hex        used for
============  =========  =================================================
:data:`PAPER`      #FAFAF8    figure, axes and savefig background
:data:`INK`        #121412    titles, axis labels, annotations
:data:`MUTED`      #5C615D    tick labels
:data:`RULE`       #CBCDCA    spines and tick marks
:data:`GRID`       #DCDEDB    gridlines
============  =========  =================================================

``GRID`` and ``RULE`` are :data:`BLUEPRINT` (#747974) mixed over paper at 22% and
35%. Two steps rather than one: the frame should read before the ruling does, and
a single flat grey for both makes a chart look boxed in.

The chrome is otherwise neutral, and deliberately: :data:`ACCENT` is the only hue
in it, so anything drawn in that blue reads as *the thing being pointed at*
rather than as decoration. It is what a single-series chart is drawn in, and what
an unlabelled PCA scatter uses.

One caveat if you reach for :data:`ACCENT` in text. Against paper it measures
3.99:1, which clears the 3:1 that a line, a marker or a rule needs but not the
4.5:1 that body text does. The docs therefore darken it one step for link text
and keep #367FC9 for marks; do the same for an annotation you expect to be read
rather than seen.

The rest of the palette does not follow the brand and should not. Lightness alone
cannot separate four overlaid curves, and taking hue out would give up the ΔE
guarantees above, so :data:`BLUE_ORANGE` and viridis stay exactly as they are.
"""

from __future__ import annotations

from contextlib import contextmanager

#: Fraction of the viridis ramp to sample. Trimmed at both ends, where the
#: colormap is nearly black and nearly white.
VIRIDIS_SPAN = (0.06, 0.94)

#: Maximum number of categorical series viridis can separate acceptably.
MAX_CATEGORICAL = 5

#: Wong's colourblind-safe blue/orange family: blue, orange, sky blue,
#: vermillion. Used for curves, where two to four series need to be told apart
#: at a glance. Passes every palette check in light and dark mode.
BLUE_ORANGE = ("#0072B2", "#E69F00", "#56B4E9", "#D55E00")

#: The brand accent, and the warm end of the same gradient it is taken from --
#: the accent's own channels reversed, which is why the two sit at one lightness.
#: Blue against orange is the pair dichromats separate best, and this one measures
#: normal-vision ΔE 27.1 / CVD 22.4, inside the band Wong's own adjacent pairs
#: occupy (24.6-36.2 / 24.6-30.1).
ACCENT = "#367FC9"
AMBER = "#C97F36"

#: The accent pair as a categorical palette. Two slots, and only two: a third
#: hue would have to come from somewhere other than the brand.
BRAND_PAIR = (ACCENT, AMBER)

#: name -> (colors or None for a sampled ramp, maximum series)
PALETTES: dict[str, int] = {
    "viridis": MAX_CATEGORICAL,
    "blue-orange": len(BLUE_ORANGE),
    "brand": len(BRAND_PAIR),
}

#: Palette for line charts: losses, training curves, error-vs-horizon.
CURVE_PALETTE = "blue-orange"

#: Palette for magnitude and for many-way categorical comparisons.
MAGNITUDE_PALETTE = "viridis"

#: Secondary encoding, so identity is never colour-alone.
MARKERS = ("o", "s", "^", "D", "v")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

#: Sequential colormap for magnitude (images, heatmaps, spectra).
SEQUENTIAL_CMAP = "viridis"

#: Background. kamara's warm off-white, not pure white -- a figure dropped into
#: the docs or the README sits on the same ground as the page around it.
PAPER = "#FAFAF8"

#: Text: titles, axis labels, annotations.
INK = "#121412"

#: Rules and borders at full strength. Used here only as the tint source for
#: :data:`GRID` and :data:`RULE`, which are what actually reach a figure.
BLUEPRINT = "#747974"

#: Secondary text: tick labels, colourbar labels. Recedes behind :data:`INK`.
MUTED = "#5C615D"

#: Blueprint at 22% over paper. Gridlines, which must sit under the data.
GRID = "#DCDEDB"

#: Blueprint at 35% over paper. Spines and tick marks, one step up from the grid
#: so the frame reads before the ruling does.
RULE = "#CBCDCA"

#: Brand token -> hex, for callers styling a figure by hand.
BRAND = {
    "paper": PAPER,
    "ink": INK,
    "blueprint": BLUEPRINT,
    "muted": MUTED,
    "grid": GRID,
    "rule": RULE,
    "accent": ACCENT,
    "amber": AMBER,
}


def palette_colors(n: int, palette: str = MAGNITUDE_PALETTE) -> list[str]:
    """``n`` hex colours from a named palette, in a fixed order.

    Raises:
        ValueError: if ``n`` exceeds what the palette can separate legibly.
            Generating an extra hue past that point puts an adjacent pair below
            the legibility floor; split the chart into small multiples instead.
    """
    if palette not in PALETTES:
        raise ValueError(f"unknown palette {palette!r}; choose from {sorted(PALETTES)}")
    limit = PALETTES[palette]
    if n < 1:
        raise ValueError(f"need at least one colour, got {n}")
    if n > limit:
        raise ValueError(
            f"{n} categorical series exceeds the {limit} that the {palette!r} palette "
            "can separate legibly (adjacent pairs fall below the normal-vision floor). "
            "Use small multiples, or group the smallest series into 'other'."
        )
    if palette == "blue-orange":
        return list(BLUE_ORANGE[:n])
    if palette == "brand":
        return list(BRAND_PAIR[:n])
    return viridis_colors(n)


def viridis_colors(n: int, span: tuple[float, float] = VIRIDIS_SPAN) -> list[str]:
    """``n`` hex colours sampled evenly from viridis, in a fixed order.

    Raises:
        ValueError: if ``n`` exceeds :data:`MAX_CATEGORICAL`. Adding a sixth
            hue would put an adjacent pair below the legibility floor; split the
            chart into small multiples or group the tail into "other".
    """
    import matplotlib
    from matplotlib.colors import to_hex

    if n < 1:
        raise ValueError(f"need at least one colour, got {n}")
    if n > MAX_CATEGORICAL:
        raise ValueError(
            f"{n} categorical series exceeds the {MAX_CATEGORICAL} that viridis can "
            "separate legibly (adjacent pairs fall below the normal-vision floor). "
            "Use small multiples, or group the smallest series into 'other'."
        )
    cmap = matplotlib.colormaps[SEQUENTIAL_CMAP]
    lo, hi = span
    if n == 1:
        return [to_hex(cmap(0.5 * (lo + hi)))]
    step = (hi - lo) / (n - 1)
    return [to_hex(cmap(lo + i * step)) for i in range(n)]


def series_style(index: int, n_series: int, palette: str = MAGNITUDE_PALETTE) -> dict:
    """Colour, marker and line style for series ``index`` of ``n_series``.

    Colour follows the series' identity (its index), never its rank, so
    filtering the chart never repaints the survivors.
    """
    colors = palette_colors(n_series, palette)
    return {
        "color": colors[index],
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "linewidth": 2.0,
        "markersize": 5.0,
        "markeredgecolor": PAPER,
        "markeredgewidth": 0.6,
    }


def rc_params(n_series: int = MAX_CATEGORICAL, palette: str = MAGNITUDE_PALETTE) -> dict:
    """Matplotlib rcParams for the xwm look: recessive axes, thin marks."""
    from cycler import cycler

    return {
        "axes.prop_cycle": cycler(color=palette_colors(n_series, palette)),
        "image.cmap": SEQUENTIAL_CMAP,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": RULE,
        "ytick.color": RULE,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "font.size": 10,
    }


def use_viridis(n_series: int = MAX_CATEGORICAL, palette: str = MAGNITUDE_PALETTE) -> None:
    """Apply the xwm style globally. Call once at the top of a script."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(rc_params(n_series, palette))


@contextmanager
def plot_style(n_series: int = MAX_CATEGORICAL, palette: str = MAGNITUDE_PALETTE):
    """Apply the xwm style for one block, leaving global rcParams untouched."""
    import matplotlib.pyplot as plt

    with plt.rc_context(rc_params(n_series, palette)):
        yield


@contextmanager
def viridis_style(n_series: int = MAX_CATEGORICAL):
    """Deprecated alias for :func:`plot_style`, kept for callers that used it."""
    with plot_style(n_series, MAGNITUDE_PALETTE):
        yield
