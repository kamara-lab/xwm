"""Train on a recorded dataset, and ask whether the synthetic twin agrees.

Examples 01-05 run on `xwm.data`'s generated worlds, 06-08 on a simulated
Franka. This one runs on data a real robot produced: `lerobot/pusht`, 206
episodes of 2-D pushing recorded through a teleoperated setup, in the LeRobot
layout that DROID and the Open X-Embodiment mirrors also use.

The reason to pair it with `xwm.data.PushWorld` is not decoration. Every claim
this library makes about planning is currently measured on data it generated
itself, and the honest question is how much that transfers. So the same model,
the same objective and the same evaluation are run twice -- once on the recorded
task and once on the synthetic one built to abstract it -- and the two horizon
curves are put side by side. Where they disagree is the interesting part.

What the comparison is *not*: the two tasks are not the same task. Push-T pushes
a T-shaped block towards a fixed pose with a human at the controls; PushWorld
pushes a disc towards a random goal with smoothed noise at the controls. Read the
curves as "does compounding error behave the same way", not as a benchmark.

The structure is example 04's, two stages with the encoder frozen for the
second, and that is not a stylistic choice. The first version of this script
trained the encoder jointly with the latent-prediction loss, and it collapsed:
over a 60-step control run on PushWorld, joint training reached `feature_std`
0.031 and `mean_cosine` 0.998 -- every observation mapped to nearly the same
vector -- while its training loss fell by a factor of 50. A LeJEPA-pretrained,
frozen encoder on the identical budget gave `feature_std` 0.232, `mean_cosine`
0.854, and a horizon ratio of 0.717 against the collapsed run's 1.87, which is
*worse* than predicting no change at all. A falling prediction loss is not
evidence of a working world model when the encoder is free to make prediction
trivial, so the collapse diagnostics are printed next to the ratio here rather
than left to a separate example.

What you should see depends on the budget, and that *is* the result.

At the laptop defaults the synthetic run works -- a healthy embedding
(`feature_std` 0.18, `mean_cosine` 0.93) and a horizon ratio of 0.87 falling to
0.71 -- and the recorded run does not: on identical architecture, budget and
objective, `lerobot/pusht` comes out collapsed (`feature_std` 0.025,
`mean_cosine` 0.999) with the ratio pinned at 1.00. Not a bug in the script. A
recipe tuned on the synthetic world does not transfer unchanged, and
`reg_weight = 20` (picked on sprites) is too low for a low-variance recorded
scene. It is not the resolution -- 64 px does not fix it -- and it is not the
weight alone either, which at this budget moves `mean_cosine` to 0.73 and leaves
`feature_std` at 0.008.

Scale it and the collapse goes away. On one A10 for 42 minutes
(`modal run deploy/app_recorded.py`, the `gpu-recorded` preset: all 206 episodes
at 96 px, `reg_weight` 100, 6000 steps per stage, 384-wide depth-6) the recorded
encoder reaches `feature_std` 0.815 and `mean_cosine` 0.111, and its dynamics
beat the no-op baseline by 13% at one step and 22% at eight -- the same shape as
the synthetic curve and within 0.05 of it. Four things changed at once, so it
isolates nothing; what it establishes is that the collapse is escapable. See
`docs/findings.md` for every number here.

Artifacts: the two horizon-error curves against their no-op baselines, both
training curves, a GIF of a recorded episode, the collapse diagnostics, and the
numbers as Markdown/LaTeX/CSV.

Needs the data extra, and downloads ~30 MB on first run:

    pip install "xwm[data]"
    python examples/09_recorded_data.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _common import describe_settings, report, setting, setup

import xwm

DATASET = setting("DATASET", "lerobot/pusht")
IMG_SIZE = setting("IMG_SIZE", 32)
PATCH_SIZE = setting("PATCH_SIZE", 4)
FRAMES = setting("CLIP_LEN", 9)
EPISODES = setting("EPISODES", 40)
PRETRAIN_STEPS = setting("PRETRAIN_STEPS", 300)
DYNAMICS_STEPS = setting("DYNAMICS_STEPS", 300)
BATCH = setting("BATCH", 16)
ENCODE_BATCH = setting("ENCODE_BATCH", 64)
REG_WEIGHT = setting("REG_WEIGHT", 20.0)
#: Figures are rendered from the data at its own resolution, not from the
#: downsampled tensor the model trains on. Push-T records at 96x96; showing a
#: 32px training clip upscaled 6x tells you about the resize, not about the task.
FIGURE_SIZE = setting("FIGURE_SIZE", 96)
FIGURE_FRAMES = setting("FIGURE_FRAMES", 48)
FIGURE_CANDIDATES = setting("FIGURE_CANDIDATES", 8)
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


def split(data, fraction=0.2):
    """Hold out whole *episodes*, not clips.

    Clips overlap and several come from one episode, so a random clip split
    leaks: a test clip can share frames with a training clip and the horizon
    error comes out flatteringly low. `episode` is in the loaded data precisely
    so this is possible to get right.
    """
    episodes = np.unique(data["episode"])
    n_test = max(2, int(round(len(episodes) * fraction)))
    held = set(episodes[-n_test:].tolist())
    test_mask = np.isin(data["episode"], list(held))
    keep = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and v.ndim >= 1}
    train = {k: v[~test_mask] for k, v in keep.items() if v.shape[0] == test_mask.shape[0]}
    test = {k: v[test_mask] for k, v in keep.items() if v.shape[0] == test_mask.shape[0]}
    return train, test


def pretrain_encoder(key, train):
    """Stage 1: LeJEPA on still frames pulled out of the clips.

    SIGReg is what makes this stage safe to train the encoder in: it penalises
    an anisotropic embedding distribution, so the shortcut of mapping every
    observation to one vector is no longer the cheapest way down.
    """
    frames = jnp.asarray(train["video"]).reshape(-1, 3, IMG_SIZE, IMG_SIZE)
    model = xwm.families.jepa.lejepa(
        key=key,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        n_targets=4,
        encoder_kwargs=ENC,
        predictor_kwargs={"depth": 3, "pred_dim": 64, "num_heads": 4},
        reg_weight=REG_WEIGHT,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(
                1e-3, PRETRAIN_STEPS, warmup_steps=PRETRAIN_STEPS // 10
            )
        ),
    )
    batches = xwm.data.iter_batches({"image": frames}, 32, key=key, epochs=None)
    state, history = trainer.fit(
        batches, steps=PRETRAIN_STEPS, key=key, log_every=PRETRAIN_STEPS
    )
    # Collapse is decided *here*, not in stage 2, so it gets reported here --
    # and reported before stage 2 can fail and take the answer with it.
    encoder = state.model.encoder
    embed = eqx.filter_jit(lambda i: jax.vmap(encoder)(i))
    sample = xwm.core.batched_apply(embed, frames[:512], batch_size=ENCODE_BATCH)
    health = xwm.metrics.collapse_report(sample.reshape(sample.shape[0], -1))
    return encoder, history[-1], health


def train_dynamics(key, train, encoder, action_dim, label):
    """Stage 2: action-conditioned dynamics inside a frozen latent space."""
    model = xwm.families.jepa.action_world_model(
        key=key,
        action_dim=action_dim,
        encoder=encoder,
        freeze_encoder=True,
        dynamics_kwargs=DYN,
        conditioning="both",
        teacher_forcing_weight=1.0,
        rollout_weight=1.0,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(
                1e-3, DYNAMICS_STEPS, warmup_steps=DYNAMICS_STEPS // 10
            )
        ),
    )
    batches = xwm.data.iter_batches(
        {"video": jnp.asarray(train["video"]), "action": jnp.asarray(train["action"])},
        BATCH,
        key=key,
        epochs=None,
    )
    print(
        f"  {label}: {trainer.n_trainable:,} of {model.n_params:,} params "
        "trainable (encoder frozen)"
    )
    state, history = trainer.fit(
        batches,
        steps=DYNAMICS_STEPS,
        key=key,
        log_every=DYNAMICS_STEPS // 3,
        callbacks=[xwm.training.print_metrics(keys=["loss_teacher_forcing", "loss_rollout"])],
    )
    return state.model, history


def figure_frames(world, n_frames):
    """One contiguous episode from each source, at :data:`FIGURE_SIZE`.

    Deliberately separate from the training tensors. The model trains at
    ``IMG_SIZE`` because that is what fits in the budget; a figure has no such
    constraint, and a 32px clip blown up 6x shows the interpolation rather than
    the robot. Push-T records at 96x96, so that is what gets drawn.

    Returns ``(recorded, synthetic)``, both ``(T, 3, FIGURE_SIZE, FIGURE_SIZE)``
    and the same length, so they can be tiled side by side.
    """
    # Not episode 0: whichever of the first few moves the most. A teleoperator
    # spends the opening of some episodes lining up, and a figure of a robot
    # holding still is a poor advertisement for an action-conditioned model. The
    # criterion is stated rather than eyeballed -- total frame-to-frame change.
    candidates = list(xwm.datasets.stream(DATASET, limit=FIGURE_CANDIDATES, resize=FIGURE_SIZE))
    motion = [float(np.abs(np.diff(c["video"], axis=0)).mean()) for c in candidates]
    chosen = int(np.argmax(motion))
    print(
        f"  figure episode {chosen} of {len(candidates)} "
        f"(mean frame-to-frame change {motion[chosen]:.4f}, "
        f"dullest {min(motion):.4f})"
    )
    recorded = np.asarray(candidates[chosen]["video"])
    length = min(n_frames, recorded.shape[0])
    # Evenly spaced across the whole episode rather than the first N: a Push-T
    # episode is ~125 frames and its first 48 are mostly the approach.
    picked = np.unique(np.linspace(0, recorded.shape[0] - 1, length).round()).astype(int)
    recorded = recorded[picked]

    big = xwm.data.PushWorld(FIGURE_SIZE)
    synthetic = np.asarray(scripted_push(big, len(picked) - 1, jr.PRNGKey(5)))
    del world
    return recorded, synthetic


def scripted_push(world, n_steps, key):
    """Drive the pusher to actually push, for the figure only.

    Random actions make a *correct* picture of the data the model trains on and
    a poor picture of the task: the pusher wanders off and the puck never moves,
    so the frames show three dots drifting. This is a hand-written controller --
    get behind the puck relative to the goal, then push through it -- included
    so the figure shows what the world is *for*.

    It is **not** a learned policy and no result is computed from it. The
    training tensors come from `push_sequences` and its random actions, exactly
    as the recorded side comes from a teleoperator.
    """
    state = world.reset(key)
    frames = [world.render(state)]
    solved = 0
    for step in range(n_steps):
        if bool(world.success(state)):
            # Solved: move the goal rather than holding still. The controller
            # gets there in about nine steps, so holding would leave four fifths
            # of the animation static -- and a second push shows the dynamics are
            # general rather than one lucky geometry.
            solved += 1
            angle = jr.uniform(jr.fold_in(key, step), (), minval=0.0, maxval=2.0 * jnp.pi)
            offset = jnp.stack([jnp.cos(angle), jnp.sin(angle)]) * 0.28
            state = state._replace(goal=jnp.clip(state.puck + offset, 0.12, 0.88))
        to_goal = state.goal - state.puck
        heading = to_goal / jnp.maximum(jnp.linalg.norm(to_goal), 1e-6)
        # Aim just *inside* contact on the far side of the puck from the goal, so
        # closing on the target point *is* the push.
        target = state.puck - heading * world.contact_radius * 0.85
        offset = target - state.pusher
        approach = jnp.clip(offset / world.action_scale, -1.0, 1.0)
        # Two phases, and the scaling is the whole trick. Approaching: scale by
        # distance, because a unit action moves the pusher 0.12 per step, which
        # overshoots the puck and -- since the contact model projects radially --
        # knocks it backwards. Touching: command a full push, because a
        # distance-scaled action at contact range is almost zero and the puck
        # creeps.
        touching = jnp.linalg.norm(state.puck - state.pusher) <= world.contact_radius * 1.05
        state = world.step(state, jnp.where(touching, heading, approach))
        frames.append(world.render(state))
    print(f"  scripted push reached the goal {solved} time(s) in {n_steps} steps")
    return jnp.stack(frames)


def collapse(model, data):
    """Collapse diagnostics on the held-out embeddings, and the embeddings.

    Printed beside the horizon ratio because the two are only interpretable
    together: a ratio near 1 means "the dynamics learned nothing useful", and
    only `feature_std` and `mean_cosine` say whether that is because the task is
    hard or because the representation died. The flattened embeddings come back
    too, for the spectrum plot -- the most legible form of the same diagnostic.
    """
    encode = eqx.filter_jit(lambda v: jax.vmap(model.encode_sequence)(v))
    z = encode(jnp.asarray(data["video"]))
    flat = z.reshape(z.shape[0] * z.shape[1], -1)
    return xwm.metrics.collapse_report(flat), flat


def horizon_errors(model, data):
    """Per-horizon latent L1 of a free rollout, and the no-op baseline.

    Identical to example 04's measurement, deliberately: the point of running it
    on recorded data is that the number means the same thing.
    """
    encode = eqx.filter_jit(lambda v: jax.vmap(model.encode_sequence)(v))
    z = xwm.core.batched_apply(encode, jnp.asarray(data["video"]), batch_size=ENCODE_BATCH)

    @eqx.filter_jit
    def rollouts(z, actions):
        return jax.vmap(model.imagine)(z[:, 0], actions)

    pred = rollouts(z, jnp.asarray(data["action"]))
    truth = z[:, 1:]
    static = jnp.broadcast_to(z[:, :1], truth.shape)
    err = jnp.mean(jnp.abs(pred - truth), axis=(0, 2, 3))
    base = jnp.mean(jnp.abs(static - truth), axis=(0, 2, 3))
    return np.asarray(err), np.asarray(base)


def main():
    out = setup("09_recorded_data", n_series=4)
    describe_settings(
        {
            "dataset": DATASET,
            "episodes": EPISODES,
            "clip": FRAMES,
            "size": IMG_SIZE,
            "pretrain": PRETRAIN_STEPS,
            "dynamics": DYNAMICS_STEPS,
        }
    )

    entry = xwm.datasets.describe(DATASET)
    print(f"{entry.name}: {entry.summary}")
    print(f"  {entry.episodes} episodes total, {entry.size_gb} GB, {entry.licence}")
    print(f"  cite: {entry.citation}\n")

    print(f"loading {EPISODES} of them (cached under {xwm.datasets.cache_dir()})")
    recorded = xwm.datasets.create(
        DATASET, length=FRAMES, stride=FRAMES, limit=EPISODES, resize=IMG_SIZE
    )
    action_dim = recorded["action"].shape[-1]
    print(
        f"  {recorded['video'].shape[0]} clips of {FRAMES} frames, "
        f"action_dim={action_dim}, {int(recorded['dropped'][0])} episodes too short"
    )

    # The synthetic twin, matched on clip count, clip length and resolution so
    # the only difference left is the data itself.
    world = xwm.data.PushWorld(IMG_SIZE)
    synthetic = xwm.data.push_sequences(
        jr.PRNGKey(0), recorded["video"].shape[0], FRAMES, world=world
    )
    synthetic = {k: np.asarray(v) for k, v in synthetic.items()}
    synthetic["episode"] = np.arange(synthetic["video"].shape[0])

    curves, tables, health, embeddings = {}, [], {}, {}
    for label, data, dim in (
        (f"recorded ({DATASET})", recorded, action_dim),
        ("synthetic (PushWorld)", synthetic, world.action_dim),
    ):
        print(f"\n{label}")
        train, test = split(data)
        print(f"  {train['video'].shape[0]} train / {test['video'].shape[0]} test clips")

        print("  stage 1: LeJEPA on still frames (no actions)")
        encoder, final, stage1 = pretrain_encoder(jr.PRNGKey(1), train)
        print(
            f"    loss {final['loss']:.4f}  "
            f"(pred {final['loss_pred']:.4f} + reg {final['loss_reg']:.4f})"
        )
        print(
            "    encoder health: "
            + "  ".join(f"{k}={float(v):.4f}" for k, v in stage1.items())
        )

        print("  stage 2: action-conditioned dynamics on the frozen encoder")
        model, history = train_dynamics(jr.PRNGKey(2), train, encoder, dim, label)
        err, base = horizon_errors(model, test)
        curves[label] = (err, base, history)
        health[label], embeddings[label] = collapse(model, test)
        for h in range(err.shape[0]):
            tables.append([label, h + 1, float(err[h]), float(base[h]), float(err[h] / base[h])])

    print("\nheld-out latent L1 of a free rollout, ratio to the no-op baseline:")
    print(f"  {'h':>3}  " + "  ".join(f"{label[:22]:>22}" for label in curves))
    for h in range(FRAMES - 1):
        cells = [f"{float(curves[k][0][h] / curves[k][1][h]):>22.2f}" for k in curves]
        print(f"  {h + 1:>3}  " + "  ".join(cells))
    print("\nratio < 1 means the dynamics beat assuming nothing changes; the ratio")
    print("rising with h is compounding error. Compare the *shapes*: if the")
    print("synthetic curve is flat where the recorded one climbs, then results")
    print("tuned on the synthetic world are optimistic about horizon.")

    print("\nrepresentation health on the same held-out clips:")
    for label, report_ in health.items():
        body = "  ".join(f"{k}={float(v):.4f}" for k, v in report_.items())
        print(f"  {label[:22]:>22}  {body}")
    print("\nRead these first. A ratio near 1 with feature_std near 0 and")
    print("mean_cosine near 1 is a collapsed encoder, not a hard task, and the")
    print("horizon numbers above it mean nothing -- see this file's docstring")
    print("for what that looks like when the encoder is trained jointly.")

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    horizons = list(range(1, FRAMES))
    series = {}
    for label, (err, base, _) in curves.items():
        series[label] = [float(v) for v in err]
        series[f"{label}, no-op"] = [float(v) for v in base]
    ax = xwm.plots.plot_horizon(
        horizons, series, title="Compounding error: recorded data vs its synthetic twin"
    )
    report(xwm.plots.save_figure(ax.figure, out / "horizon_error.png"))

    for label, (_, _, history) in curves.items():
        name = "recorded" if label.startswith("recorded") else "synthetic"
        ax = xwm.plots.plot_history(
            history, keys=["loss_teacher_forcing", "loss_rollout"], logy=True, title=label
        )
        report(xwm.plots.save_figure(ax.figure, out / f"training_curve_{name}.png"))

    # The spectrum is the same collapse finding as the table above, in the form
    # that needs no explaining: a healthy encoder's singular values decay
    # gently, a collapsed one's fall off a cliff after a handful of directions.
    ax = xwm.plots.plot_spectra(
        {label.split(" (")[0]: z for label, z in embeddings.items()},
        title="Embedding spectra: recorded against synthetic",
    )
    report(xwm.plots.save_figure(ax.figure, out / "spectra.png"))

    for label, z in embeddings.items():
        name = "recorded" if label.startswith("recorded") else "synthetic"
        # Coloured by position within the clip, which is meaningful for both
        # sources: an encoder that kept anything about the trajectory shows a
        # gradient, a collapsed one shows a blob.
        steps = jnp.tile(jnp.arange(FRAMES), z.shape[0] // FRAMES)
        ax = xwm.plots.plot_latent_pca(
            z, labels=steps, label_name="frame in clip", title=f"{label} embeddings"
        )
        report(xwm.plots.save_figure(ax.figure, out / f"latent_pca_{name}.png"))

    # Frames at the data's own resolution, the two sources side by side. This is
    # the comparison the whole example is about, so it gets to be one image.
    rec_frames, syn_frames = figure_frames(world, FIGURE_FRAMES)
    labels = [f"recorded -- {DATASET}", "synthetic -- xwm.data.PushWorld"]
    tiled = xwm.plots.tile_frames(
        [rec_frames, syn_frames], labels=labels, scale=2, label_height=20
    )
    report(xwm.plots.save_gif(out / "episode.gif", tiled, fps=8))
    # tile_frames builds (T, H, W, 3) uint8 for save_gif; plot_frames wants
    # (T, C, H, W) in [0, 1].
    strip = xwm.plots.tile_frames(
        [rec_frames, syn_frames], labels=labels, scale=3, label_height=26
    )
    fig = xwm.plots.plot_frames(
        strip.transpose(0, 3, 1, 2).astype(np.float32) / 255.0,
        max_frames=6,
        scale=2.2,
        suptitle="One episode from each source, at the resolution it was recorded",
    )
    report(xwm.plots.save_figure(fig, out / "episode_frames.png", dpi=200))

    report(
        xwm.plots.save_table(
            out / "collapse",
            ["source", *next(iter(health.values())).keys()],
            [[label, *[float(v) for v in r.values()]] for label, r in health.items()],
            caption=(
                "Collapse diagnostics on the held-out clips. feature_std near 0 "
                "with mean_cosine near 1 means the horizon numbers are measuring "
                "a dead representation."
            ),
            label="tab:collapse",
        )
    )
    report(
        xwm.plots.save_table(
            out / "horizon_error",
            ["source", "horizon", "model L1", "no-op L1", "ratio"],
            tables,
            caption=(
                f"Held-out latent L1 of a free rollout on {EPISODES} episodes of "
                f"{DATASET} and on a matched sample of xwm.data.PushWorld. Ratio "
                "below 1 beats assuming nothing changes."
            ),
            label="tab:recorded",
        )
    )


if __name__ == "__main__":
    main()
