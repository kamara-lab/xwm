"""Pretrain I-JEPA on images and inspect the resulting representation.

The loss going down proves nothing on its own: a collapsing encoder drives its
prediction loss down too, because it is predicting its own degenerate output.
So this script reports two other things.

**A linear probe** for the agent's ground-truth position. Be warned that on
this toy world it saturates -- the position is nearly a linear function of the
red channel, and `ridge_probe` standardises features before fitting, so it
recovers the target even from a badly degraded embedding. A high R2 here is not
evidence of a healthy representation. It is included precisely to show that.

**Collapse diagnostics**, which are the informative signal: `feature_std` -> 0
and `mean_cosine` -> 1 both mean collapse. Expect to see exactly that here --
2048 images of a blob over a few hundred steps is far too easy, and the constant
solution is an easy win. That is not a bug in I-JEPA or in this implementation;
it is what the EMA-teacher mechanism costs, and it is why LeJEPA exists.
`examples/03_collapse_strategies.py` runs the comparison.

Artifacts: training curves, the multi-block mask being sampled (PNG + GIF), the
embedding spectrum, a latent PCA coloured by true agent position, and the
results as Markdown/LaTeX/CSV.

Run: python examples/01_image_ijepa.py
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from _common import report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 32)
PATCH_SIZE = setting("PATCH_SIZE", 4)
STEPS = setting("STEPS", 300)
BATCH = setting("BATCH", 32)
ENCODE_BATCH = setting("ENCODE_BATCH", 64)
ENC = {
    "depth": setting("ENC_DEPTH", 4),
    "embed_dim": setting("ENC_DIM", 128),
    "num_heads": setting("ENC_HEADS", 4),
}


def probe(model, data_train, data_eval):
    """Ridge-probe the frozen encoder for the ground-truth agent position."""
    embed = jax.jit(jax.vmap(model.embed))
    z_train = xwm.core.batched_apply(embed, data_train["image"], batch_size=ENCODE_BATCH)
    z_eval = xwm.core.batched_apply(embed, data_eval["image"], batch_size=ENCODE_BATCH)
    scores = xwm.metrics.ridge_probe(
        z_train, data_train["position"], z_eval, data_eval["position"]
    )
    return scores, z_eval


def main():
    out = setup("01_image_ijepa")
    key = jr.PRNGKey(0)
    k_data, k_model, k_train = jr.split(key, 3)

    world = xwm.data.SpriteWorld(IMG_SIZE, n_distractors=2)
    train = xwm.data.sprite_images(jr.fold_in(k_data, 0), 2048, world=world)
    evaluation = xwm.data.sprite_images(jr.fold_in(k_data, 1), 512, world=world)

    model = xwm.families.jepa.ijepa(
        key=k_model,
        size="tiny",
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        n_targets=4,
        target_scale=0.15,
        encoder_kwargs=ENC,
        predictor_kwargs={"depth": 3, "pred_dim": 64, "num_heads": 4},
    )
    print(xwm.tools.summary(model, max_depth=1))
    sampler = model.mask_sampler
    print(
        f"grid {model.encoder.grid}  context {sampler.context_size} tokens  "
        f"targets {sampler.n_targets}x{sampler.target_size}\n"
    )

    before, _ = probe(model, train, evaluation)
    print(f"probe R2 before training: {float(before['r2']):.3f}")

    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(xwm.training.cosine_warmup(1e-3, STEPS, warmup_steps=STEPS // 10)),
        ema_momentum=xwm.training.ema_momentum(STEPS, base=0.996),
    )
    batches = xwm.data.iter_batches(train, BATCH, key=k_train, epochs=None)
    state, history = trainer.fit(
        batches,
        steps=STEPS,
        key=k_train,
        log_every=10,
        callbacks=[xwm.training.print_metrics(every=5, keys=["loss", "embed_std"])],
    )

    after, z_eval = probe(state.model, train, evaluation)
    report_metrics = xwm.metrics.collapse_report(z_eval)
    print(f"\nprobe R2 after training:  {float(after['r2']):.3f}  (saturated -- see docstring)")
    print(f"loss {history[0]['loss']:.4f} -> {history[-1]['loss']:.4f}")
    print(f"\ncollapse diagnostics (embed_dim {state.model.encoder.embed_dim}):")
    for name, value in report_metrics.items():
        print(f"  {name:<12} {float(value):.4f}")
    if float(report_metrics["mean_cosine"]) > 0.9:
        print(
            "\nmean_cosine is near 1: the encoder has largely collapsed, despite the\n"
            "prediction loss looking excellent. Run examples/03_collapse_strategies.py\n"
            "to see what SIGReg does to this same setup."
        )

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    # Two panels rather than two y-axes: the loss and the embedding spread live
    # on different scales, and a dual-axis chart would invite false comparison.
    # (With collapse="ema" the total loss and the prediction loss are the same
    # number, so only one of them is worth drawing.)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xwm.plots.plot_history(
        history, keys=["loss_pred"], logy=True, ax=axes[0], title="Prediction loss"
    )
    xwm.plots.plot_history(
        history,
        keys=["embed_std"],
        ax=axes[1],
        ylabel="embedding std",
        title="Embedding spread (collapse indicator)",
    )
    fig.tight_layout()
    report(xwm.plots.save_figure(fig, out / "training_curve.png"))

    ax = xwm.plots.plot_spectrum(z_eval, title="Embedding spectrum after I-JEPA")
    report(xwm.plots.save_figure(ax.figure, out / "spectrum.png"))

    ax = xwm.plots.plot_latent_pca(
        z_eval,
        labels=evaluation["position"][:, 0],
        label_name="true agent x",
        title="Latent PCA, coloured by true position",
    )
    report(xwm.plots.save_figure(ax.figure, out / "latent_pca.png"))

    # What the model is actually asked to predict.
    frame = evaluation["image"][0]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for i, cell in enumerate(axes):
        mask = sampler(jr.fold_in(k_data, 100 + i))
        xwm.plots.plot_mask(frame, mask.context, model.encoder.grid, ax=cell)
    fig.suptitle("Four draws of the I-JEPA multi-block mask (visible tokens bright)")
    fig.tight_layout()
    report(xwm.plots.save_figure(fig, out / "masks.png"))

    # ...and the same thing as an animation, one frame per mask draw.
    masked_frames = []
    for i in range(24):
        mask = sampler(jr.fold_in(k_data, 200 + i))
        visible = jnp.zeros(sampler.n_tokens, bool).at[mask.context].set(True)
        gh, gw = model.encoder.grid
        alpha = jnp.kron(
            visible.reshape(gh, gw).astype(jnp.float32),
            jnp.ones((IMG_SIZE // gh, IMG_SIZE // gw)),
        )
        masked_frames.append(frame * (0.25 + 0.75 * alpha)[None])
    report(xwm.plots.save_gif(out / "mask_draws.gif", jnp.stack(masked_frames), fps=4, scale=6))

    report(
        xwm.plots.save_metrics(
            out / "metrics",
            {
                "steps": STEPS,
                "image_size": IMG_SIZE,
                "embed_dim": state.model.encoder.embed_dim,
                "probe_r2_before": float(before["r2"]),
                "probe_r2_after": float(after["r2"]),
                "loss_first": history[0]["loss"],
                "loss_last": history[-1]["loss"],
                **{k: float(v) for k, v in report_metrics.items()},
            },
        )
    )

    rows = [
        ["probe R2 (before)", float(before["r2"])],
        ["probe R2 (after)", float(after["r2"])],
        ["prediction loss (first)", history[0]["loss"]],
        ["prediction loss (last)", history[-1]["loss"]],
        *[[name, float(value)] for name, value in report_metrics.items()],
    ]
    report(
        xwm.plots.save_table(
            out / "results",
            ["quantity", "value"],
            rows,
            caption=f"I-JEPA on {IMG_SIZE}x{IMG_SIZE} sprite images, {STEPS} steps.",
            label="tab:ijepa",
        )
    )


if __name__ == "__main__":
    main()
