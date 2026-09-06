"""Plot, GIF and table export: artifacts must be produced without a display."""

import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("PIL")

import xwm  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# -- viridis style ----------------------------------------------------------
def test_viridis_colors_are_stable_and_ordered():
    """Colour follows identity, so the first N colours must not change with N."""
    four = xwm.plots.viridis_colors(4)
    assert len(four) == 4
    assert all(c.startswith("#") for c in four)
    assert xwm.plots.viridis_colors(4) == four  # deterministic
    assert xwm.plots.viridis_colors(2)[0] == four[0]  # first slot is fixed


def test_viridis_refuses_too_many_categorical_series():
    """Beyond 5 slots an adjacent pair drops below the legibility floor."""
    xwm.plots.viridis_colors(xwm.plots.MAX_CATEGORICAL)
    with pytest.raises(ValueError, match="small multiples"):
        xwm.plots.viridis_colors(xwm.plots.MAX_CATEGORICAL + 1)
    with pytest.raises(ValueError, match="at least one"):
        xwm.plots.viridis_colors(0)


def test_series_style_adds_secondary_encoding():
    """Identity must never rest on hue alone."""
    styles = [xwm.plots.series_style(i, 4) for i in range(4)]
    assert len({s["color"] for s in styles}) == 4
    assert len({s["marker"] for s in styles}) == 4
    assert len({str(s["linestyle"]) for s in styles}) == 4


def test_rc_params_use_viridis_sequentially():
    rc = xwm.plots.rc_params(3)
    assert rc["image.cmap"] == "viridis"
    assert len(rc["axes.prop_cycle"]) == 3


def _oklab(rgb):
    """sRGB -> OKLab, the space every separation number here is measured in."""
    rgb = np.asarray(rgb)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    return m2 @ np.cbrt(m1 @ lin)


def test_chrome_comes_from_the_kamara_tokens():
    """Everything that is not data is brand surface, not matplotlib defaults."""
    rc = xwm.plots.rc_params(3)
    assert rc["figure.facecolor"] == xwm.plots.PAPER == "#FAFAF8"
    assert rc["savefig.facecolor"] == xwm.plots.PAPER
    assert rc["axes.facecolor"] == xwm.plots.PAPER
    assert rc["text.color"] == rc["axes.labelcolor"] == xwm.plots.INK == "#121412"
    assert rc["xtick.labelcolor"] == rc["ytick.labelcolor"] == xwm.plots.MUTED
    assert rc["grid.color"] == xwm.plots.GRID
    assert rc["axes.edgecolor"] == xwm.plots.RULE
    # Markers punch out of the paper ground, not a white one that would ring
    # them once the figure lands on the page.
    assert xwm.plots.series_style(0, 2)["markeredgecolor"] == xwm.plots.PAPER


def test_chrome_is_recessive_in_the_right_order():
    """Grid under rule under text: three steps, not one flat grey."""
    from matplotlib.colors import to_rgb

    lightness = [sum(to_rgb(c)) / 3 for c in (xwm.plots.INK, xwm.plots.MUTED,
                                              xwm.plots.RULE, xwm.plots.GRID,
                                              xwm.plots.PAPER)]
    assert lightness == sorted(lightness), "ink -> paper must increase monotonically"


def test_the_brand_pair_is_the_accent_and_its_reversal():
    """The two accents are each other's channels reversed, so they share a
    lightness -- and the pair stops at two, because a third hue would have to
    come from outside the brand."""
    assert xwm.plots.ACCENT == "#367FC9"
    assert xwm.plots.AMBER == "#C97F36"
    assert xwm.plots.AMBER[1:] == "".join(
        reversed([xwm.plots.ACCENT[1:][i : i + 2] for i in range(0, 6, 2)])
    )
    assert xwm.plots.palette_colors(2, "brand") == ["#367FC9", "#C97F36"]
    with pytest.raises(ValueError, match="exceeds the 2"):
        xwm.plots.palette_colors(3, "brand")


def test_the_brand_pair_clears_the_separation_floor():
    """The accent pair is held to the same measured floor as every other
    palette, not waved through because it is the brand's."""
    from matplotlib.colors import to_rgb

    a, b = (_oklab(to_rgb(c)) for c in xwm.plots.BRAND_PAIR)
    assert float(np.linalg.norm(a - b)) * 100 > 15.0


def test_series_colours_do_not_collapse_to_the_neutral_brand():
    """Lightness alone cannot separate four overlaid curves, so the rebrand
    stops at the chrome and the measured-ΔE palettes are left intact."""
    from matplotlib.colors import to_rgb

    for hexcolor in xwm.plots.palette_colors(4, "blue-orange"):
        r, g, b = to_rgb(hexcolor)
        assert max(r, g, b) - min(r, g, b) > 0.2, f"{hexcolor} is greyscale"


def test_viridis_style_context_is_scoped():
    import matplotlib.pyplot as plt

    before = plt.rcParams["image.cmap"]
    with xwm.plots.viridis_style(2):
        assert plt.rcParams["image.cmap"] == "viridis"
    assert plt.rcParams["image.cmap"] == before


# -- figures ----------------------------------------------------------------
def test_plot_history_draws_a_legend_for_multiple_series(key):
    history = [{"step": i, "loss": 1.0 / i, "loss_pred": 0.9 / i} for i in range(1, 12)]
    ax = xwm.plots.plot_history(history, logy=True, title="t")
    assert len(ax.lines) == 2
    assert ax.get_legend() is not None
    single = xwm.plots.plot_history(history, keys=["loss"])
    assert single.get_legend() is None  # one series needs no legend box


def test_plot_history_validates_input():
    with pytest.raises(ValueError, match="empty"):
        xwm.plots.plot_history([])
    with pytest.raises(ValueError, match="no metrics"):
        xwm.plots.plot_history([{"step": 1}])


def test_plot_spectrum_and_spectra(key):
    z = jr.normal(key, (128, 32))
    assert "RankMe" in xwm.plots.plot_spectrum(z).get_title()
    ax = xwm.plots.plot_spectra({"a": z, "b": 0.1 * z})
    assert len(ax.lines) == 2 and ax.get_legend() is not None


def test_plot_bars_and_horizon():
    ax = xwm.plots.plot_bars(["a", "b", "c"], [1.0, 2.0, 3.0], ylabel="metric")
    assert len(ax.patches) == 3
    with pytest.raises(ValueError, match="values"):
        xwm.plots.plot_bars(["a"], [1.0, 2.0])
    ax = xwm.plots.plot_horizon([1, 2, 3], {"model": [1.0, 2.0, 3.0], "baseline": [2.0, 3.0, 4.0]})
    assert len(ax.lines) == 2


def test_frame_figures(key):
    world = xwm.data.SpriteWorld(16, n_distractors=1)
    frames, _ = world.rollout(key, jnp.zeros((4, 2)))
    assert xwm.plots.plot_frames(frames, suptitle="s") is not None
    assert xwm.plots.plot_rollout(frames) is not None
    assert xwm.plots.plot_rollout(frames, frames) is not None


def test_plot_mask_handles_rgb_and_grayscale(key):
    world = xwm.data.SpriteWorld(16, n_distractors=0)
    frame = world.render(world.reset(key))
    mask = xwm.masking.MultiBlockMask2d((4, 4), n_targets=2, target_scale=0.2)(key)
    assert "tokens visible" in xwm.plots.plot_mask(frame, mask.context, (4, 4)).get_title()
    assert xwm.plots.plot_mask(frame[:1], mask.context, (4, 4)) is not None


def test_plot_latent_pca(key):
    z = jr.normal(key, (64, 16))
    assert xwm.plots.plot_latent_pca(z) is not None
    assert xwm.plots.plot_latent_pca(z, labels=jr.uniform(key, (64,)), label_name="x") is not None


def test_save_figure(key, tmp_path):
    ax = xwm.plots.plot_spectrum(jr.normal(key, (32, 8)))
    path = xwm.plots.save_figure(ax.figure, tmp_path / "sub" / "fig.png")
    assert path.exists() and path.stat().st_size > 1000


# -- GIFs -------------------------------------------------------------------
def test_frames_to_uint8_accepts_all_layouts(key):
    chw = np.asarray(jr.uniform(key, (5, 3, 8, 8)))
    assert xwm.plots.frames_to_uint8(chw).shape == (5, 8, 8, 3)
    assert xwm.plots.frames_to_uint8(chw.transpose(0, 2, 3, 1)).shape == (5, 8, 8, 3)
    assert xwm.plots.frames_to_uint8(chw[:, 0]).shape == (5, 8, 8, 3)  # grayscale
    out = xwm.plots.frames_to_uint8(chw)
    assert out.dtype == np.uint8 and out.max() <= 255


def test_frames_to_uint8_rejects_bad_shapes():
    with pytest.raises(ValueError, match="3-D or 4-D"):
        xwm.plots.frames_to_uint8(np.zeros((4,)))
    with pytest.raises(ValueError, match="channel axis"):
        xwm.plots.frames_to_uint8(np.zeros((5, 7, 8, 9)))


def test_upscale_is_nearest_neighbour():
    frames = np.arange(4, dtype=np.uint8).reshape(1, 2, 2, 1).repeat(3, axis=-1)
    up = xwm.plots.upscale(frames, 3)
    assert up.shape == (1, 6, 6, 3)
    assert np.array_equal(up[0, :3, :3, 0], np.full((3, 3), frames[0, 0, 0, 0]))
    assert xwm.plots.upscale(frames, 1) is frames


def test_save_gif_single_and_tiled(key, tmp_path):
    from PIL import Image

    world = xwm.data.SpriteWorld(16, n_distractors=1)
    a, _ = world.rollout(key, jr.normal(key, (6, 2)))
    b, _ = world.rollout(jr.PRNGKey(1), jr.normal(jr.PRNGKey(1), (6, 2)))

    single = xwm.plots.save_gif(tmp_path / "a.gif", a, fps=6, scale=3)
    with Image.open(single) as im:
        assert im.n_frames == 7 and im.size == (48, 48)

    tiled = xwm.plots.save_gif(tmp_path / "b.gif", [a, b], labels=["real", "imagined"], scale=3)
    with Image.open(tiled) as im:
        assert im.n_frames == 7
        assert im.size[0] > 90  # two panels side by side
        assert im.size[1] > 48  # plus a label strip


def test_tile_frames_truncates_to_shortest(key):
    a = np.asarray(jr.uniform(key, (8, 3, 6, 6)))
    b = np.asarray(jr.uniform(key, (5, 3, 6, 6)))
    assert xwm.plots.tile_frames([a, b], scale=2).shape[0] == 5


def test_tile_frames_validates_labels(key):
    a = np.asarray(jr.uniform(key, (3, 3, 4, 4)))
    with pytest.raises(ValueError, match="labels"):
        xwm.plots.tile_frames([a, a], labels=["only-one"])


# -- tables -----------------------------------------------------------------
def test_markdown_table_is_aligned():
    md = xwm.plots.markdown_table(["name", "value"], [["ema", 0.002], ["sigreg", 0.08]])
    lines = md.splitlines()
    assert len(lines) == 4
    assert len({len(line) for line in lines}) == 1  # every row the same width
    assert "0.0020" in md


def test_latex_table_escapes_and_wraps():
    tex = xwm.plots.latex_table(
        ["feature_std", "100%"], [["a_b", 1.5]], caption="A & B", label="tab:x"
    )
    assert r"\_" in tex and r"\%" in tex and r"\&" in tex
    assert r"\begin{tabular}{lr}" in tex
    assert r"\toprule" in tex and r"\bottomrule" in tex
    assert r"\caption{A \& B}" in tex and r"\label{tab:x}" in tex
    bare = xwm.plots.latex_table(["a", "b"], [[1, 2]])
    assert r"\begin{table}" not in bare  # no caption/label -> tabular only


def test_latex_table_validates_align():
    with pytest.raises(ValueError, match="align"):
        xwm.plots.latex_table(["a", "b"], [[1, 2]], align="lrr")


def test_latex_table_without_booktabs():
    tex = xwm.plots.latex_table(["a"], [[1]], booktabs=False)
    assert r"\hline" in tex and r"\toprule" not in tex


def test_format_cell():
    assert xwm.plots.format_cell(1.5) == "1.5000"
    assert xwm.plots.format_cell(1.5, "{:.1f}") == "1.5"
    assert xwm.plots.format_cell(True) == "yes"
    assert xwm.plots.format_cell("x") == "x"
    assert xwm.plots.format_cell(512) == "512"


def test_format_cell_unwraps_numpy_and_jax_scalars():
    """A training loop hands over device scalars, not Python floats."""
    assert xwm.plots.format_cell(np.float32(0.99912345)) == "0.9991"
    assert xwm.plots.format_cell(jnp.float32(13.98)) == "13.9800"
    assert xwm.plots.format_cell(np.int64(512)) == "512"
    assert xwm.plots.format_cell(np.bool_(True)) == "yes"


def test_save_table_writes_json_and_latex(tmp_path):
    written = xwm.plots.save_table(
        tmp_path / "results" / "collapse",
        ["strategy", "mean_cosine"],
        [["ema", 0.999], ["sigreg", 0.022]],
        caption="Collapse",
        label="tab:collapse",
    )
    assert set(written) == {"json", "tex"}
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0
    assert r"\toprule" in written["tex"].read_text()

    payload = json.loads(written["json"].read_text())
    assert payload["columns"] == ["strategy", "mean_cosine"]
    assert payload["rows"][0] == {"strategy": "ema", "mean_cosine": 0.999}
    assert payload["caption"] == "Collapse" and payload["label"] == "tab:collapse"


def test_save_table_json_keeps_full_precision(tmp_path):
    """Rounding belongs to display, not storage -- .tex rounds, .json must not."""
    value = 0.123456789
    written = xwm.plots.save_table(tmp_path / "t", ["x"], [[value]])
    assert json.loads(written["json"].read_text())["rows"][0]["x"] == value
    assert "0.1235" in written["tex"].read_text()


def test_save_table_no_longer_writes_markdown_or_csv(tmp_path):
    written = xwm.plots.save_table(tmp_path / "t", ["x"], [[1.0]])
    assert not (tmp_path / "t.md").exists()
    assert not (tmp_path / "t.csv").exists()
    assert set(written) == {"json", "tex"}


def test_save_metrics_writes_flat_json(tmp_path):
    path = xwm.plots.save_metrics(
        tmp_path / "m", {"r2": np.float64(0.2951), "steps": 512, "name": "franka"}
    )
    assert path.name == "m.json"
    payload = json.loads(path.read_text())
    assert payload == {"r2": 0.2951, "steps": 512, "name": "franka"}


def test_json_handles_non_finite_and_arrays(tmp_path):
    """JSON has no NaN literal; emitting one makes the file unparseable."""
    path = xwm.plots.save_json(
        tmp_path / "j",
        {"nan": float("nan"), "inf": float("inf"), "arr": np.arange(3), "nested": {"a": 1.5}},
    )
    payload = json.loads(path.read_text())  # would raise if NaN were emitted raw
    assert payload["nan"] is None and payload["inf"] is None
    assert payload["arr"] == [0, 1, 2]
    assert payload["nested"] == {"a": 1.5}


def test_jsonable_scalars():
    assert xwm.plots.jsonable(np.float32(1.5)) == pytest.approx(1.5)
    assert xwm.plots.jsonable(jnp.arange(2)) == [0, 1]
    assert xwm.plots.jsonable(True) is True
    assert xwm.plots.jsonable(None) is None


def test_markdown_table_is_still_available_for_printing():
    """Not written to disk any more, but still useful for a terminal."""
    md = xwm.plots.markdown_table(["a"], [[1.0]])
    assert md.startswith("| a")


def test_plot_latent_pca_rejects_mismatched_labels(key):
    """(B, N, D) flattens to B*N points, which rarely matches B labels."""
    z = jr.normal(key, (8, 16, 32))
    with pytest.raises(ValueError, match="Pool token sequences"):
        xwm.plots.plot_latent_pca(z, labels=jnp.arange(8))
    # Pooled to one vector per sample, it works.
    assert xwm.plots.plot_latent_pca(jnp.mean(z, axis=-2), labels=jnp.arange(8)) is not None


# -- palettes ---------------------------------------------------------------
def test_blue_orange_is_the_curve_palette():
    """Curves default to blue-orange; magnitude encodings stay on viridis."""
    assert xwm.plots.CURVE_PALETTE == "blue-orange"
    assert xwm.plots.MAGNITUDE_PALETTE == "viridis"
    assert xwm.plots.palette_colors(2, "blue-orange") == ["#0072B2", "#E69F00"]


def test_palette_colors_are_stable_across_series_counts():
    """Colour follows identity: the first N colours must not shift with N."""
    four = xwm.plots.palette_colors(4, "blue-orange")
    assert xwm.plots.palette_colors(2, "blue-orange") == four[:2]
    assert xwm.plots.palette_colors(3, "blue-orange") == four[:3]


def test_each_palette_enforces_its_own_limit():
    """Blue-orange has four usable slots; viridis five. Past that, refuse."""
    xwm.plots.palette_colors(4, "blue-orange")
    with pytest.raises(ValueError, match="small multiples"):
        xwm.plots.palette_colors(5, "blue-orange")
    xwm.plots.palette_colors(5, "viridis")
    with pytest.raises(ValueError, match="small multiples"):
        xwm.plots.palette_colors(6, "viridis")
    with pytest.raises(ValueError, match="unknown palette"):
        xwm.plots.palette_colors(2, "nope")


def test_loss_curves_use_blue_then_orange():
    history = [{"step": i, "loss": 1.0 / i, "loss_pred": 0.9 / i} for i in range(1, 10)]
    ax = xwm.plots.plot_history(history)
    assert [line.get_color() for line in ax.lines] == ["#0072B2", "#E69F00"]


def test_horizon_curves_use_blue_then_orange():
    ax = xwm.plots.plot_horizon([1, 2], {"model": [1.0, 2.0], "baseline": [2.0, 3.0]})
    assert [line.get_color() for line in ax.lines] == ["#0072B2", "#E69F00"]


def test_bars_stay_on_viridis():
    """Five-way comparisons need viridis; blue-orange cannot separate them."""
    ax = xwm.plots.plot_bars(list("abcde"), [1.0, 2.0, 3.0, 4.0, 5.0])
    colors = {patch.get_facecolor() for patch in ax.patches}
    assert len(colors) == 5


def test_palette_override_is_respected():
    history = [{"step": i, "loss": 1.0 / i} for i in range(1, 5)]
    ax = xwm.plots.plot_history(history, palette="viridis")
    assert ax.lines[0].get_color() == xwm.plots.palette_colors(1, "viridis")[0]


def test_gif_uses_one_palette_for_every_frame(key, tmp_path):
    """Per-frame palettes make an animation shimmer even on static pixels."""
    from PIL import Image

    world = xwm.data.SpriteWorld(16, n_distractors=1)
    frames, _ = world.rollout(key, jr.normal(key, (6, 2)))
    path = xwm.plots.save_gif(tmp_path / "a.gif", frames)
    with Image.open(path) as im:
        palettes = set()
        for index in range(im.n_frames):
            im.seek(index)
            palette = im.getpalette()
            # A frame with no local palette inherits the global one, which is
            # exactly what a shared palette produces.
            palettes.add(tuple(palette) if palette else None)
        assert len([p for p in palettes if p is not None]) == 1


def test_gif_validates_palette_size(key, tmp_path):
    frames = jr.uniform(key, (3, 3, 8, 8))
    with pytest.raises(ValueError, match=r"colors must be in \[2, 256\]"):
        xwm.plots.save_gif(tmp_path / "a.gif", frames, colors=300)


def test_gif_scale_defaults_to_native(key, tmp_path):
    """The default must not silently upscale: blocks are not detail."""
    from PIL import Image

    frames = jr.uniform(key, (3, 3, 24, 24))
    with Image.open(xwm.plots.save_gif(tmp_path / "a.gif", frames)) as im:
        assert im.size == (24, 24)
    with Image.open(xwm.plots.save_gif(tmp_path / "b.gif", frames, scale=3)) as im:
        assert im.size == (72, 72)
