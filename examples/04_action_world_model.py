"""Learn action-conditioned dynamics on a frozen encoder, V-JEPA 2-AC style.

Two stages, and the split is the whole idea:

1. Learn a representation from passive observation, with no actions at all.
2. Freeze it, and learn action-conditioned dynamics *inside* that latent space.

Stage 2 is cheap -- it trains only the dynamics model -- and it is well-posed
precisely because the encoder is frozen: the prediction targets are fixed
functions of the observations, so there is no way to make the loss smaller by
degrading the representation.

The evaluation measures compounding error: one-step prediction is easy, and the
question that matters for planning is how fast accuracy decays over a horizon.

Artifacts: the horizon-error curve against the no-op baseline, training curves
for both loss terms, a GIF of a held-out episode, and the per-horizon numbers as
Markdown/LaTeX/CSV.

Run: python examples/04_action_world_model.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from _common import report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 32)
PATCH_SIZE = setting("PATCH_SIZE", 4)
FRAMES = setting("CLIP_LEN", 9)
PRETRAIN_STEPS = setting("PRETRAIN_STEPS", 300)
DYNAMICS_STEPS = setting("DYNAMICS_STEPS", 400)
ENCODE_BATCH = setting("ENCODE_BATCH", 64)
ENC = {
    "depth": setting("ENC_DEPTH", 4),
    "embed_dim": setting("ENC_DIM", 128),
    "num_heads": setting("ENC_HEADS", 4),
}
DYN = {
    "depth": setting("DYN_DEPTH", 4),
    "pred_dim": setting("DYN_DIM", 128),
    "num_heads": setting("DYN_HEADS", 4),
}


def pretrain_encoder(key, images):
    """Stage 1: LeJEPA on still frames -- no actions involved."""
    model = xwm.families.jepa.lejepa(
        key=key, img_size=IMG_SIZE, patch_size=PATCH_SIZE, n_targets=4,
        encoder_kwargs=ENC, predictor_kwargs={"depth": 3, "pred_dim": 64, "num_heads": 4},
        reg_weight=20.0,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(1e-3, PRETRAIN_STEPS, warmup_steps=PRETRAIN_STEPS // 10)
        ),
    )
    batches = xwm.data.iter_batches({"image": images}, 32, key=key, epochs=None)
    state, history = trainer.fit(batches, steps=PRETRAIN_STEPS, key=key, log_every=PRETRAIN_STEPS)
    return state.model.encoder, history[-1]


def horizon_errors(model, data):
    """Per-horizon latent L1 error of a free rollout, plus a no-op baseline.

    The baseline predicts "nothing changes". A dynamics model that cannot beat
    it has learned nothing about the actions.
    """
    # vmap inside the jitted function, not around the bound method: wrapping a
    # jax-transformed callable would hide the model's parameters from Equinox.
    encode = eqx.filter_jit(lambda v: jax.vmap(model.encode_sequence)(v))
    # Chunked: this is a dataset, not a minibatch.
    z = xwm.core.batched_apply(encode, data["video"], batch_size=ENCODE_BATCH)

    @eqx.filter_jit
    def rollouts(z, actions):
        return jax.vmap(model.imagine)(z[:, 0], actions)

    pred = rollouts(z, data["action"])  # (B, T-1, N, D)
    truth = z[:, 1:]
    static = jnp.broadcast_to(z[:, :1], truth.shape)  # "nothing changes"
    err = jnp.mean(jnp.abs(pred - truth), axis=(0, 2, 3))
    base = jnp.mean(jnp.abs(static - truth), axis=(0, 2, 3))
    return err, base


def main():
    out = setup("04_action_world_model", n_series=2)
    key = jr.PRNGKey(0)
    k_data, k_enc, k_dyn = jr.split(key, 3)
    world = xwm.data.SpriteWorld(IMG_SIZE, n_distractors=2)

    images = xwm.data.sprite_images(jr.fold_in(k_data, 0), 2048, world=world)["image"]
    train = xwm.data.sprite_sequences(jr.fold_in(k_data, 1), 512, FRAMES, world=world)
    test = xwm.data.sprite_sequences(jr.fold_in(k_data, 2), 128, FRAMES, world=world)

    print("stage 1: pretraining the encoder on still frames (no actions)")
    encoder, final = pretrain_encoder(k_enc, images)
    print(
        f"  loss {final['loss']:.4f}  "
        f"(pred {final['loss_pred']:.4f} + reg {final['loss_reg']:.4f})"
    )

    print("\nstage 2: action-conditioned dynamics on the frozen encoder")
    model = xwm.families.jepa.action_world_model(
        key=k_dyn,
        action_dim=world.action_dim,
        encoder=encoder,
        freeze_encoder=True,
        conditioning="both",
        dynamics_kwargs=DYN,
        teacher_forcing_weight=1.0,
        rollout_weight=1.0,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(1e-3, DYNAMICS_STEPS, warmup_steps=DYNAMICS_STEPS // 10)
        ),
    )
    print(f"  trainable {trainer.n_trainable:,} of {model.n_params:,} params (encoder frozen)")
    batches = xwm.data.iter_batches(
        {"video": train["video"], "action": train["action"]}, 16, key=k_dyn, epochs=None
    )
    state, history = trainer.fit(
        batches, steps=DYNAMICS_STEPS, key=k_dyn, log_every=DYNAMICS_STEPS // 4,
        callbacks=[xwm.training.print_metrics(keys=["loss_teacher_forcing", "loss_rollout"])],
    )

    err, base = horizon_errors(state.model, test)
    print("\nheld-out latent L1 error of a free rollout, by horizon:")
    print(f"  {'h':>3}  {'model':>8}  {'no-op':>8}  {'ratio':>6}")
    for h in range(err.shape[0]):
        print(f"  {h + 1:>3}  {float(err[h]):>8.4f}  {float(base[h]):>8.4f}  "
              f"{float(err[h] / base[h]):>6.2f}")
    print("\nratio < 1 means the dynamics beat assuming nothing changes;")
    print("the ratio rising with h is compounding error, and it is what")
    print("bounds a useful planning horizon.")
    print("\nNote the ratio at h=1: one step barely changes the image, so")
    print("'predict no change' is a strong short-horizon baseline -- which is")
    print("why a one-step loss is a poor way to judge a world model.")

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    horizons = list(range(1, err.shape[0] + 1))
    ax = xwm.plots.plot_horizon(
        horizons,
        {"learned dynamics": [float(v) for v in err],
         "no-op baseline": [float(v) for v in base]},
        title="Compounding error: free rollout vs 'nothing changes'",
    )
    report(xwm.plots.save_figure(ax.figure, out / "horizon_error.png"))

    ax = xwm.plots.plot_history(
        history,
        keys=["loss_teacher_forcing", "loss_rollout"],
        logy=True,
        title="Action-conditioned dynamics (encoder frozen)",
    )
    report(xwm.plots.save_figure(ax.figure, out / "training_curve.png"))

    clip = test["video"][0]
    report(xwm.plots.save_gif(out / "episode.gif", clip, fps=5, scale=6))
    fig = xwm.plots.plot_frames(clip, suptitle="A held-out episode (actions drive the red agent)")
    report(xwm.plots.save_figure(fig, out / "episode_frames.png"))

    rows = [
        [h, float(err[h - 1]), float(base[h - 1]), float(err[h - 1] / base[h - 1])]
        for h in horizons
    ]
    report(
        xwm.plots.save_table(
            out / "horizon_error",
            ["horizon", "model L1", "no-op L1", "ratio"],
            rows,
            caption=(
                "Held-out latent L1 error of a free rollout on a frozen encoder. "
                "Ratio below 1 beats assuming nothing changes."
            ),
            label="tab:horizon",
        )
    )


if __name__ == "__main__":
    main()
