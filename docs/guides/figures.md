# Figures and tables

Examples write to `examples/outputs/<name>/`: `*.png` figures, `*.gif` animations,
`*.json` metrics at full precision, and `*.tex` tables of the same numbers
formatted for a paper.

```bash
pip install "xwm[plots]"
```

```python
xwm.plots.save_table(out / "results", headers, rows, caption=..., label=...)
xwm.plots.save_metrics(out / "metrics", {"probe_r2": 0.295})
xwm.plots.save_gif(out / "rollout.gif", [real, imagined], labels=["real", "imagined"])
xwm.plots.save_figure(ax.figure, out / "curve.png")
```

**Rounding is a display concern**, so the `.tex` rounds and the `.json` does not.
Every example writes both, from the same numbers: the table is for reading, the
JSON is for comparing runs.

## Palettes, and their measured limits

Three named palettes, chosen by what the chart is doing:

| palette | for | colours |
| --- | --- | --- |
| `"blue-orange"` (`CURVE_PALETTE`) | **curves**: losses, histories, error-vs-horizon | Wong's colourblind-safe blue, orange, sky blue, vermillion |
| `"viridis"` (`MAGNITUDE_PALETTE`) | **magnitude**: images, PCA colourbars, many-way bars | perceptually uniform |
| `"brand"` | **two-way**: real vs imagined, before vs after | the accent pair, `#367FC9` and `#C97F36` |

Line charts usually carry two to four series and need maximum separation between
them; the Wong family delivers adjacent-pair ΔE of 24–36 in OKLab, passing every
check in both light and dark mode. A continuous quantity wants a perceptually
uniform ramp instead.

Using viridis for *categorical* series means sampling discrete steps from a
sequential ramp. That works, but only up to a point, and the point is **measurable
rather than a matter of taste**:

| slots | CVD separation ΔE | normal-vision ΔE |
| --- | --- | --- |
| 2 | 62.8 | 63.7 |
| 3 | 28.3 | 33.0 |
| 4 | 18.6 | 22.4 |
| 5 | 13.6 | 16.3 |
| 6 | 10.4 | **13.2, too low** |

A normal-vision ΔE below 15 means readers with full colour vision cannot reliably
tell the pair apart, and no amount of secondary encoding fixes that. So
[`viridis_colors`](../reference/plots.md#xwm.plots.viridis_colors) **refuses** more
than `MAX_CATEGORICAL = 5` series and tells you to use small multiples instead. It
is a limit that is much more useful enforced than documented.

```python
xwm.plots.viridis_colors(6)
# ValueError: 6 categorical series exceeds the 5 that viridis can separate ...
```

Two further consequences of using a sequential ramp categorically, both handled:

- Its **ends** are very dark and very light, so the extreme steps have low
  contrast against the plot surface. Every figure therefore carries a legend, and
  every example also writes the same numbers as a table, so nothing depends on
  reading a colour.
- Its **middle** is desaturated and reads grey-ish. Series get distinct markers and
  line styles as well as colours, so identity never rests on hue alone.

### The accent pair

`"brand"` is `ACCENT` (`#367FC9`) and `AMBER` (`#C97F36`), both read off the
kamara gradient. They are each other's channels reversed, which is why they sit
at one lightness, and blue against orange is the pair dichromats separate best.
Held to the same measurement as everything else here:

| pair | normal-vision ΔE | CVD ΔE |
| --- | --- | --- |
| `#367FC9` / `#C97F36` | 27.1 | 22.4 |
| Wong's own adjacent pairs, for scale | 27.1–36.2 | 24.6–30.1 |

It stops at two slots, and asking for a third raises, because a third hue would
have to come from outside the brand.

## Chrome

Everything that is *not* data — background, text, gridlines, spines, tick marks —
comes from the [kamara design tokens](https://kamara.dev), the same ones this site
is built from, so a figure sits on the same ground as the page around it instead
of announcing itself as a white rectangle:

| token | hex | used for |
| --- | --- | --- |
| `PAPER` | `#FAFAF8` | figure, axes and savefig background |
| `INK` | `#121412` | titles, axis labels, annotations |
| `MUTED` | `#5C615D` | tick labels |
| `RULE` | `#CBCDCA` | spines and tick marks |
| `GRID` | `#DCDEDB` | gridlines |
| `ACCENT` | `#367FC9` | the one hue in the chrome: single-series charts, unlabelled scatters |

`GRID` and `RULE` are `BLUEPRINT` (`#747974`) mixed over paper at 22% and 35%.
Two steps rather than one: the frame should read before the ruling does, and a
single flat grey for both makes a chart look boxed in. All six are exported from
`xwm.plots`, and collected in `xwm.plots.BRAND`, if you are styling a figure by
hand.

`ACCENT` is the only hue in the chrome, and deliberately so: anything drawn in
that blue reads as *the thing being pointed at* rather than as decoration.

!!! warning "Accent in text"

    `#367FC9` measures 3.99:1 against paper. That clears the 3:1 a line, marker
    or rule needs, but not the 4.5:1 body text needs. These docs darken it one
    step (`#2769AD`, 5.42:1) for link text and keep `#367FC9` for marks — do the
    same for an annotation you expect to be read rather than seen.

**The rest of the palette does not follow the brand, and should not.** Lightness
alone cannot separate four overlaid curves, and taking hue out would give up the
ΔE guarantees above, so `blue-orange` and viridis stay exactly as they are.

## Style

```python
with xwm.plots.plot_style(n_series=4, palette="blue-orange"):
    fig, ax = plt.subplots()
    ...

xwm.plots.save_figure(fig, out / "curve.png")
```

[`plot_style`](../reference/plots.md#xwm.plots.plot_style) is a context manager over
matplotlib's rcParams; [`series_style(i, n)`](../reference/plots.md#xwm.plots.series_style)
returns the colour, marker and linestyle for series `i` if you are drawing by hand.

## Ready-made plots

| function | draws |
| --- | --- |
| [`plot_history`](../reference/plots.md#xwm.plots.plot_history) | training curves from `Trainer.fit`'s history |
| [`plot_horizon`](../reference/plots.md#xwm.plots.plot_horizon) | error against rollout horizon, with a baseline |
| [`plot_spectrum`](../reference/plots.md#xwm.plots.plot_spectrum) / [`plot_spectra`](../reference/plots.md#xwm.plots.plot_spectra) | singular values of an embedding matrix |
| [`plot_latent_pca`](../reference/plots.md#xwm.plots.plot_latent_pca) | latents in 2-D, coloured by a ground-truth quantity |
| [`plot_mask`](../reference/plots.md#xwm.plots.plot_mask) | a context/target split over the token grid |
| [`plot_rollout`](../reference/plots.md#xwm.plots.plot_rollout) | real against imagined frames |
| [`plot_frames`](../reference/plots.md#xwm.plots.plot_frames) | a strip of frames |
| [`plot_bars`](../reference/plots.md#xwm.plots.plot_bars) | a categorical comparison |

```python
fig, ax = plt.subplots()
xwm.plots.plot_history(history, keys=["loss", "embed_std"], ax=ax, logy=True)
```

<figure markdown="span">
  ![Latent PCA coloured by true position](../outputs/01_image_ijepa/latent_pca.png){ width="560" }
  <figcaption>
    <code>plot_latent_pca</code>: embeddings in two dimensions, coloured by the
    ground-truth quantity a probe would try to recover.
  </figcaption>
</figure>

## Tables

```python
paths = xwm.plots.save_table(
    out / "results",
    ["policy", "mean (m)", "% of gap closed"],
    rows,
    caption="Franka tool distance to a goal image after 5 steps.",
    label="tab:franka-planning",
    float_format="{:.4f}",
)
# {'json': .../results.json, 'tex': .../results.tex}
```

One call writes both formats from one set of rows, so the LaTeX in a paper and the
JSON a script compares can never disagree: the `.tex` rounds to `float_format`,
the `.json` keeps full precision.
[`table_to_dict`](../reference/plots.md#xwm.plots.table_to_dict) reads one back;
[`latex_table`](../reference/plots.md#xwm.plots.latex_table) and
[`markdown_table`](../reference/plots.md#xwm.plots.markdown_table) render to a
string without writing, which is how the tables in these docs were produced.

## Animations

```python
xwm.plots.save_gif(
    out / "rollout.gif", [real, imagined],
    labels=["real", "imagined"], fps=8, scale=3,
)
```

Frames are `(T, 3, H, W)` floats or uint8; a *list* of stacks is tiled side by side
with labels. [`frames_to_uint8`](../reference/plots.md#xwm.plots.frames_to_uint8),
[`tile_frames`](../reference/plots.md#xwm.plots.tile_frames) and
[`upscale`](../reference/plots.md#xwm.plots.upscale) are the pieces, if you want to
assemble something else.

## Matplotlib is optional

`xwm.plots` imports matplotlib lazily, so the core library installs and trains
without it. The functions that need it raise a clear error rather than failing at
import time, which is why `plots` is an extra and not a dependency.
