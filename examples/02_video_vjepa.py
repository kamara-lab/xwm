"""Pretrain V-JEPA on video with tube masking.

The interesting part is what tube masking asks of the model. A tube spans the
whole clip, so the masked content is absent from *every* visible frame -- the
model cannot copy it from a neighbour and has to infer it from spatial context
and motion. Compare the two mask presets below: long-range tubes remove far
more of each frame and are correspondingly harder.

Artifacts: loss curves for both presets, a GIF of a clip, a figure showing which
tokens each preset hides frame by frame, and the comparison as a table.

Run: python examples/02_video_vjepa.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from _common import report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 32)
PATCH_SIZE = setting("PATCH_SIZE", 4)
FRAMES = setting("CLIP_LEN", 8)
STEPS = setting("STEPS", 200)
BATCH = setting("BATCH", 8)
ENC = {
    "depth": setting("ENC_DEPTH", 4),
    "embed_dim": setting("ENC_DIM", 128),
    "num_heads": setting("ENC_HEADS", 4),
}
PRESETS = {"short-range": (8, 0.15), "long-range": (2, 0.40)}


def build(key, n_targets, spatial_scale):
    return xwm.families.jepa.vjepa(
        key=key,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_frames=FRAMES,
        tubelet_size=2,
        n_targets=n_targets,
        spatial_scale=spatial_scale,
        encoder_kwargs=ENC,
        predictor_kwargs={"depth": 3, "pred_dim": 64, "num_heads": 4},
    )


def tube_overlay(clip, mask, grid, image_size):
    """Dim the tokens a tube mask hides, frame by frame."""
    gt, gh, gw = grid
    visible = jnp.zeros(gt * gh * gw, bool).at[mask.context].set(True).reshape(gt, gh, gw)
    tubelet = clip.shape[0] // gt
    alpha = jnp.repeat(
        jnp.kron(
            visible.astype(jnp.float32),
            jnp.ones((1, image_size // gh, image_size // gw)),
        ),
        tubelet,
        axis=0,
    )
    return clip * (0.25 + 0.75 * alpha)[:, None]


def main():
    out = setup("02_video_vjepa", n_series=2)
    key = jr.PRNGKey(0)
    k_data, k_model, k_train = jr.split(key, 3)
    world = xwm.data.SpriteWorld(IMG_SIZE, n_distractors=2)
    data = xwm.data.sprite_sequences(k_data, 512, FRAMES, world=world)
    print(f"clips {tuple(data['video'].shape)}\n")

    histories, rows, spectra = {}, [], {}
    for preset, (n_targets, spatial_scale) in PRESETS.items():
        model = build(k_model, n_targets, spatial_scale)
        sampler, grid = model.mask_sampler, model.encoder.grid
        print(
            f"{preset}: grid {grid} = {sampler.n_tokens} tokens | "
            f"{sampler.n_targets} tubes x {sampler.target_size} tokens "
            f"(spanning {sampler.temporal_extent}/{grid[0]} steps) | "
            f"context {sampler.context_size}"
        )
        trainer = xwm.training.Trainer(
            model,
            xwm.training.adamw(xwm.training.cosine_warmup(1e-3, STEPS, warmup_steps=STEPS // 10)),
            ema_momentum=xwm.training.ema_momentum(STEPS, base=0.996),
        )
        batches = xwm.data.iter_batches(
            {"video": data["video"]}, BATCH, key=k_train, epochs=None
        )
        state, history = trainer.fit(batches, steps=STEPS, key=k_train, log_every=10)
        histories[preset] = history

        z = jax.jit(jax.vmap(state.model.encoder.eval_mode()))(data["video"][:64])
        spectra[preset] = z.reshape(-1, z.shape[-1])
        diagnostics = xwm.metrics.collapse_report(z)
        print(f"  loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")
        print("  " + "  ".join(f"{k}={float(v):.3f}" for k, v in diagnostics.items()) + "\n")
        rows.append(
            [
                preset,
                sampler.n_targets,
                sampler.target_size,
                sampler.context_size,
                history[-1]["loss"],
                float(diagnostics["feature_std"]),
                float(diagnostics["mean_cosine"]),
                float(diagnostics["rankme"]),
            ]
        )

    # -- artifacts ----------------------------------------------------------
    print("artifacts:")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, (preset, history) in enumerate(histories.items()):
        style = xwm.plots.series_style(i, len(histories), xwm.plots.CURVE_PALETTE)
        every = max(1, len(history) // 12)
        ax.plot(
            [r["step"] for r in history],
            [r["loss"] for r in history],
            label=preset,
            markevery=every,
            **style,
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("prediction loss")
    ax.set_title("V-JEPA: short- vs long-range tube masking")
    ax.legend()
    report(xwm.plots.save_figure(fig, out / "training_curves.png"))

    ax = xwm.plots.plot_spectra(spectra, title="Embedding spectra by mask preset")
    report(xwm.plots.save_figure(ax.figure, out / "spectra.png"))

    clip = data["video"][0]
    report(xwm.plots.save_gif(out / "clip.gif", clip, fps=6, scale=6))

    overlays, labels = [clip], ["observed"]
    for preset, (n_targets, spatial_scale) in PRESETS.items():
        model = build(k_model, n_targets, spatial_scale)
        mask = model.mask_sampler(jr.fold_in(k_data, 11))
        overlays.append(tube_overlay(clip, mask, model.encoder.grid, IMG_SIZE))
        labels.append(preset)
    report(xwm.plots.save_gif(out / "tube_masks.gif", overlays, labels=labels, fps=4, scale=5))

    fig = xwm.plots.plot_frames(
        overlays[1], suptitle="Short-range tubes: hidden tokens stay hidden across the clip"
    )
    report(xwm.plots.save_figure(fig, out / "tube_mask_frames.png"))

    report(
        xwm.plots.save_table(
            out / "results",
            ["preset", "tubes", "tokens/tube", "context", "loss", "feature_std",
             "mean_cosine", "rankme"],
            rows,
            caption=(
                f"V-JEPA tube-masking presets on {IMG_SIZE}x{IMG_SIZE}x{FRAMES} clips, "
                f"{STEPS} steps."
            ),
            label="tab:vjepa",
        )
    )


if __name__ == "__main__":
    main()
