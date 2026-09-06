"""Figures, animations and result tables.

Three palettes, picked by what the chart does: **blue-orange** (Wong's
colourblind-safe family) for curves -- losses, training histories, error by
horizon -- **viridis** for magnitude and for many-way categorical comparisons,
and **brand** for the two-way comparison that should look like the rest of the
site. :mod:`xwm.plots.style` records the measured separation of each and enforces
its series limit rather than documenting it.

Everything around the data -- background, text, gridlines, spines -- comes from
the kamara design tokens (https://kamara.dev): :data:`PAPER`, :data:`INK`,
:data:`MUTED`, :data:`RULE`, :data:`GRID` and the :data:`ACCENT` blue, collected
in :data:`BRAND`.

Matplotlib and Pillow are optional dependencies (``pip install xwm[plots]``);
importing this module without them succeeds, and the error surfaces only when a
helper that needs them is actually called.
"""

from .curves import plot_bars, plot_history, plot_horizon, plot_spectra, plot_spectrum
from .export import frames_to_uint8, save_figure, save_gif, tile_frames, upscale
from .style import (
    ACCENT,
    AMBER,
    BLUE_ORANGE,
    BLUEPRINT,
    BRAND,
    BRAND_PAIR,
    CURVE_PALETTE,
    GRID,
    INK,
    MAGNITUDE_PALETTE,
    MAX_CATEGORICAL,
    MUTED,
    PALETTES,
    PAPER,
    RULE,
    SEQUENTIAL_CMAP,
    VIRIDIS_SPAN,
    palette_colors,
    plot_style,
    rc_params,
    series_style,
    use_viridis,
    viridis_colors,
    viridis_style,
)
from .tables import (
    escape_latex,
    format_cell,
    jsonable,
    latex_table,
    markdown_table,
    save_json,
    save_metrics,
    save_table,
    table_to_dict,
)
from .visualize import plot_frames, plot_latent_pca, plot_mask, plot_rollout

__all__ = [
    "ACCENT",
    "AMBER",
    "BLUEPRINT",
    "BLUE_ORANGE",
    "BRAND",
    "BRAND_PAIR",
    "GRID",
    "INK",
    "MUTED",
    "PAPER",
    "RULE",
    "CURVE_PALETTE",
    "MAGNITUDE_PALETTE",
    "MAX_CATEGORICAL",
    "PALETTES",
    "SEQUENTIAL_CMAP",
    "VIRIDIS_SPAN",
    "escape_latex",
    "format_cell",
    "frames_to_uint8",
    "jsonable",
    "latex_table",
    "markdown_table",
    "palette_colors",
    "plot_bars",
    "plot_frames",
    "plot_history",
    "plot_horizon",
    "plot_latent_pca",
    "plot_mask",
    "plot_rollout",
    "plot_spectra",
    "plot_spectrum",
    "plot_style",
    "rc_params",
    "save_figure",
    "save_gif",
    "save_json",
    "save_metrics",
    "save_table",
    "series_style",
    "table_to_dict",
    "tile_frames",
    "upscale",
    "use_viridis",
    "viridis_colors",
    "viridis_style",
]
