"""A world model for a Franka arm, simulated in Newton.

Everything up to here used a toy world whose dynamics are linear and fully
observable. That is convenient but it cannot answer the question the JEPA family
exists for, because in a toy world there is nothing unpredictable to discard.
This example runs the same pipeline against a real robot simulator:

* **Robot**: Franka Emika FR3, 7 actuated joints, loaded from URDF.
* **Physics**: `Newton <https://github.com/newton-physics/newton>`_ (NVIDIA
  Warp), Featherstone articulated-body solver.
* **Observations**: 64x64 RGB from Newton's Warp raytracer -- which runs on
  CPU, so this works headless and without a GPU.
* **Actions**: 7-dimensional joint-target deltas.

Why this is a harder problem than the sprite world: the map from joint angles
to image is nonlinear and many-to-one, links occlude each other, and the arm has
inertia, so a single frame does not determine the state. A predictive model has
to represent what the actions can control and ignore the rest.

The pipeline is unchanged from examples 04 and 05 -- which is the point. The
environment supplies the same ``{"video", "action"}`` batches that
`xwm.data.sprite_sequences` does, so nothing in `xwm.action` or `xwm.planning`
knows the difference.

**Read the result before reusing this as a recipe: at this training budget the
planner does not beat doing nothing.** The rest of the pipeline is fine -- the
dynamics beat a no-op baseline in latent space (ratio ~0.79 at horizon 7), and
the latent cost is informative (picking the lowest-cost candidate from real
rollouts lands near the best available). What fails is planning *through* the
learned model:

* With no action penalty the planner takes large actions and ends ~12% *further*
  from the goal than not moving at all -- it is exploiting model error, the
  classic failure of optimising against an imperfect world model.
* With an action penalty it stops moving instead (mean |a| falls from 0.32 to
  0.04) and merely reproduces the no-op baseline.

So the penalty does not fix the planner, it silences it. Both numbers are
printed below rather than the flattering one. The bottleneck is dynamics
accuracy: ~500 episodes and ~900 optimizer steps on a 7-DoF arm from 64x64
pixels is orders of magnitude less than the V-JEPA 2-AC setting this imitates.
Compare `examples/05_planning.py`, where the same code closes 44% of the gap on
a task whose dynamics are actually learnable at this scale.

Needs the Newton extra::

    pip install "xwm[newton]"

The first run downloads the Franka asset (~a few MB) and compiles Warp kernels.

Artifacts: episode GIFs, horizon-error curves, a joint-angle probe, and tables.

Run: python examples/06_franka_newton.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from _common import describe_settings, report, setting, setup

import xwm

# Defaults are laptop-CPU sized. A launcher (see deploy/modal_app.py) raises
# them through XWM_* environment variables for a GPU run, so there is only one
# copy of the pipeline to keep correct.
IMG_SIZE = setting("IMG_SIZE", 64)
PATCH_SIZE = setting("PATCH_SIZE", 8)
CLIP_LEN = setting("CLIP_LEN", 8)
N_TRAIN, N_TEST = setting("N_TRAIN", 512), setting("N_TEST", 96)
PRETRAIN_STEPS = setting("PRETRAIN_STEPS", 700)
DYNAMICS_STEPS = setting("DYNAMICS_STEPS", 900)
BATCH = setting("BATCH", 32)
DYN_BATCH = setting("DYN_BATCH", 16)
PLAN_HORIZON = setting("PLAN_HORIZON", 5)
PLAN_EPISODES = setting("PLAN_EPISODES", 12)
PLAN_SAMPLES = setting("PLAN_SAMPLES", 256)
#: Chunk size for encoding a whole dataset (probes, held-out rollouts).
ENCODE_BATCH = setting("ENCODE_BATCH", 64)
#: Resolution for figures and animations, independent of what the model sees.
PREVIEW_SIZE = setting("PREVIEW_SIZE", 384)
# Path-traced figures. RTX = 0 turns them off; otherwise they are written when
# OVRTX is installed and a GPU is present, and skipped with a note when not.
RTX = setting("RTX", 1)
RTX_SIZE = setting("RTX_SIZE", 512)
# Supersampling for the fallback path, where every output pixel costs
# RTX_SAMPLES ** 2 rays.
RTX_SAMPLES = setting("RTX_SAMPLES", 3)
FIGURE_DPI = setting("FIGURE_DPI", 200)
SHADOWS = setting("SHADOWS", False)
TEXTURES = setting("TEXTURES", False)
SOLVER = setting("SOLVER", "auto")
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


def collect(env, n_sequences, length, *, seed, tag):
    print(f"  collecting {n_sequences} {tag} episodes of {length} frames...", flush=True)
    data = xwm.envs.franka_sequences(env, n_sequences, length, seed=seed, smoothness=0.75)
    print(f"    video {data['video'].shape}  action {data['action'].shape}", flush=True)
    return data


def pretrain_encoder(key, frames):
    """Stage 1: LeJEPA on individual frames -- no actions involved."""
    model = xwm.families.jepa.lejepa(
        key=key,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        n_targets=4,
        encoder_kwargs=ENC,
        predictor_kwargs={"depth": 3, "pred_dim": 64, "num_heads": 4},
        reg_weight=20.0,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(1e-3, PRETRAIN_STEPS, warmup_steps=PRETRAIN_STEPS // 10)
        ),
    )
    batches = xwm.data.iter_batches({"image": frames}, BATCH, key=key, epochs=None)
    state, history = trainer.fit(
        batches, steps=PRETRAIN_STEPS, key=key, log_every=PRETRAIN_STEPS // 5
    )
    return state.model.encoder, history


def train_dynamics(key, encoder, data, action_dim):
    """Stage 2: action-conditioned latent dynamics on the frozen encoder."""
    model = xwm.families.jepa.action_world_model(
        key=key,
        action_dim=action_dim,
        encoder=encoder,
        freeze_encoder=True,
        dynamics_kwargs=DYN,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(
            xwm.training.cosine_warmup(1e-3, DYNAMICS_STEPS, warmup_steps=DYNAMICS_STEPS // 10)
        ),
    )
    print(f"  trainable {trainer.n_trainable:,} of {model.n_params:,} params (encoder frozen)")
    batches = xwm.data.iter_batches(
        {"video": jnp.asarray(data["video"]), "action": jnp.asarray(data["action"])},
        DYN_BATCH,
        key=key,
        epochs=None,
    )
    state, history = trainer.fit(
        batches, steps=DYNAMICS_STEPS, key=key, log_every=DYNAMICS_STEPS // 5
    )
    return state.model, history


def horizon_errors(model, data):
    """Per-horizon latent L1 error of a free rollout, against a no-op baseline."""
    encode = eqx.filter_jit(lambda v: jax.vmap(model.encode_sequence)(v))
    # Chunked: the held-out set is a dataset, not a minibatch, and encoding it
    # whole asks the allocator for tens of gigabytes.
    z = xwm.core.batched_apply(encode, jnp.asarray(data["video"]), batch_size=ENCODE_BATCH)

    @eqx.filter_jit
    def rollouts(z, actions):
        return jax.vmap(model.imagine)(z[:, 0], actions)

    pred = rollouts(z, jnp.asarray(data["action"]))
    truth = z[:, 1:]
    static = jnp.broadcast_to(z[:, :1], truth.shape)
    return (
        jnp.mean(jnp.abs(pred - truth), axis=(0, 2, 3)),
        jnp.mean(jnp.abs(static - truth), axis=(0, 2, 3)),
        z,
    )


def probe_joints(model, train, test):
    """Can a linear probe read the joint angles out of the frozen embedding?

    Unlike the sprite world's position probe, this one does not saturate: joint
    angles are a genuinely nonlinear function of the image.
    """
    embed = eqx.filter_jit(lambda v: jax.vmap(model.embed)(v))

    def flat(d):
        frames = jnp.asarray(d["video"]).reshape(-1, *d["video"].shape[2:])
        # 2048 episodes x 8 frames at 128px is ~16k images: embed in chunks.
        return (
            xwm.core.batched_apply(embed, frames, batch_size=ENCODE_BATCH),
            jnp.asarray(d["joint_q"]).reshape(-1, d["joint_q"].shape[-1]),
        )

    z_train, y_train = flat(train)
    z_test, y_test = flat(test)
    return xwm.metrics.ridge_probe(z_train, y_train, z_test, y_test)


#: Action penalties to compare. 0.0 shows raw model exploitation; a small
#: penalty shows the planner falling silent instead. Reporting both is the
#: point -- a single number here would be misleading either way.
ACTION_PENALTIES = (0.0, 0.05)


def plan_episodes(env, model, key):
    """Reach a goal *image* by planning in latent space, then measure in metres.

    Returns ``(results, action_magnitudes, strips, traces)``. The action magnitudes are
    what distinguish "the planner found nothing useful to do" from "the planner
    moved and made things worse".
    """
    dynamics = model.dynamics_fn()
    encode = eqx.filter_jit(model.encode)
    planner = xwm.planning.CEM(
        PLAN_HORIZON,
        env.action_dim,
        n_samples=PLAN_SAMPLES,
        n_elites=max(8, PLAN_SAMPLES // 8),
        n_iters=6,
        init_std=0.6,
    )

    @eqx.filter_jit
    def plan(k, z0, z_goal, action_penalty):
        cost = xwm.planning.goal_cost(z_goal, kind="l2", action_penalty=action_penalty)
        return planner.plan(k, dynamics, z0, cost)

    labels = {p: f"planner (a_pen={p:g})" for p in ACTION_PENALTIES}
    traces = None
    results = {"no-op": [], "random": [], "oracle": [], **{labels[p]: [] for p in labels}}
    magnitudes = {labels[p]: [] for p in labels}
    strips = None

    for i in range(PLAN_EPISODES):
        rng = np.random.default_rng(1000 + i)
        seed = int(rng.integers(0, 2**31 - 1))
        oracle = xwm.envs.smooth_actions(rng, 1, PLAN_HORIZON, env.action_dim, smoothness=0.5)[0]
        random = xwm.envs.smooth_actions(rng, 1, PLAN_HORIZON, env.action_dim, smoothness=0.5)[0]

        goal = env.rollout(oracle, seed=seed)
        goal_tool, goal_frame = goal["tool"][-1], goal["video"][-1]
        start_frame, start_tool = goal["video"][0], goal["tool"][0]
        z0, z_goal = encode(jnp.asarray(start_frame)), encode(jnp.asarray(goal_frame))

        random_run = env.rollout(random, seed=seed)
        results["no-op"].append(float(np.linalg.norm(start_tool - goal_tool)))
        results["random"].append(float(np.linalg.norm(random_run["tool"][-1] - goal_tool)))
        results["oracle"].append(0.0)  # the oracle sequence generated the goal

        planned_runs, planned_actions = {}, {}
        for penalty in ACTION_PENALTIES:
            actions = np.asarray(plan(jr.fold_in(key, i), z0, z_goal, penalty).actions)
            planned_actions[penalty] = actions
            run = env.rollout(actions, seed=seed)
            results[labels[penalty]].append(float(np.linalg.norm(run["tool"][-1] - goal_tool)))
            magnitudes[labels[penalty]].append(float(np.abs(actions).mean()))
            planned_runs[penalty] = run

        if i == 0:
            # Seed and actions, not just pixels: the physics is deterministic
            # given them, so the episode can be replayed later through a
            # different renderer and still be the *same* episode.
            traces = {
                "seed": seed,
                "planner": planned_actions[ACTION_PENALTIES[0]],
                "oracle": oracle,
                "random": random,
            }
            strips = {
                "goal": np.broadcast_to(goal_frame[None], goal["video"].shape).copy(),
                "planner": planned_runs[ACTION_PENALTIES[0]]["video"],
                "oracle": goal["video"],
                "random": random_run["video"],
            }
    return results, magnitudes, strips, traces


def path_traced(env, actions, *, seed, size):
    """Replay one episode through the OVRTX path tracer.

    The Warp raytracer that produces observations casts one ray per pixel: hard
    shadows, flat ambient, no global illumination. That is the right trade for
    the ~500k frames a training run consumes and the wrong one for a figure.
    Newton's OVRTX viewer costs seconds per frame instead of milliseconds and
    returns soft shadows, ambient occlusion, real materials and studio lighting.

    The replay is driven by the seed and the action sequence rather than by
    stored pixels, so the path-traced episode is the same episode the numbers
    were computed from -- not a lookalike re-rolled from a different state.
    """
    with env.high_quality_renderer(backend="rtx", size=(size, size)) as renderer:
        env.reset(seed=seed)
        renderer.add(env.state)
        for action in actions:
            env.step(action)
            renderer.add(env.state)
        return renderer.frames


def main():
    out = setup("06_franka_newton", n_series=4)
    key = jr.PRNGKey(0)
    k_enc, k_dyn, k_plan = jr.split(key, 3)

    describe_settings(
        {
            "image": IMG_SIZE,
            "patch": PATCH_SIZE,
            "episodes": f"{N_TRAIN}/{N_TEST}",
            "steps": f"{PRETRAIN_STEPS}+{DYNAMICS_STEPS}",
            "encoder": ENC,
            "shadows": SHADOWS,
            "textures": TEXTURES,
            "solver": SOLVER,
        }
    )
    print("building the Newton environment (Franka FR3, Warp raytracer)")
    env = xwm.envs.FrankaEnv(
        xwm.envs.FrankaConfig(
            image_size=IMG_SIZE,
            enable_shadows=SHADOWS,
            enable_textures=TEXTURES,
            solver=SOLVER,
        )
    )
    print(
        f"  {env.action_dim} actuated joints | observations {env.observation_shape} | "
        f"solver '{env.solver_name}' | tool body "
        f"'{list(env.model.body_label)[env.tool_body_index]}'"
    )

    train = collect(env, N_TRAIN, CLIP_LEN, seed=0, tag="training")
    test = collect(env, N_TEST, CLIP_LEN, seed=1, tag="held-out")
    frames = jnp.asarray(train["video"]).reshape(-1, *train["video"].shape[2:])

    print("\nstage 1: LeJEPA on Franka frames (no actions)")
    encoder, enc_history = pretrain_encoder(k_enc, frames)
    print(f"  loss {enc_history[0]['loss']:.4f} -> {enc_history[-1]['loss']:.4f}")

    print("\nstage 2: action-conditioned dynamics on the frozen encoder")
    model, dyn_history = train_dynamics(k_dyn, encoder, train, env.action_dim)
    print(
        f"  teacher-forcing {dyn_history[-1]['loss_teacher_forcing']:.4f}  "
        f"rollout {dyn_history[-1]['loss_rollout']:.4f}"
    )

    err, base, z_test = horizon_errors(model, test)
    print("\nheld-out latent L1 error of a free rollout, by horizon:")
    print(f"  {'h':>3}  {'model':>8}  {'no-op':>8}  {'ratio':>6}")
    for h in range(err.shape[0]):
        print(f"  {h + 1:>3}  {float(err[h]):>8.4f}  {float(base[h]):>8.4f}  "
              f"{float(err[h] / base[h]):>6.2f}")

    probe = probe_joints(model, train, test)
    print(f"\nlinear probe for the 7 joint angles: R2 = {float(probe['r2']):.3f}")

    print(f"\nplanning to goal images with CEM over {PLAN_EPISODES} episodes")
    plans, magnitudes, strips, traces = plan_episodes(env, model, k_plan)
    gap = float(np.mean(plans["no-op"]))
    if gap <= 1e-9:
        raise RuntimeError(
            "the no-op baseline is zero, so start and goal coincide -- the episode "
            "generator is broken and every policy would look perfect"
        )
    closed = {p: 100.0 * (1.0 - float(np.mean(plans[p])) / gap) for p in plans}
    policies = ["no-op", "random", *[f"planner (a_pen={p:g})" for p in ACTION_PENALTIES], "oracle"]

    print("\ntrue tool distance to the goal (metres, lower is better):")
    print(f"  {'policy':<22}{'mean':>8}{'median':>8}{'worst':>8}{'%gap':>7}{'mean |a|':>10}")
    for policy in policies:
        v = np.asarray(plans[policy])
        mag = magnitudes.get(policy)
        mag_text = f"{float(np.mean(mag)):>10.3f}" if mag else " " * 10
        print(
            f"  {policy:<22}{v.mean():>8.3f}{np.median(v):>8.3f}{v.max():>8.3f}"
            f"{closed[policy]:>+7.0f}{mag_text}"
        )

    # Read the verdict off the numbers rather than hardcoding it: the result
    # depends on how well the dynamics were fit, and it changed sign between the
    # laptop-scale and GPU-scale runs. A conclusion printed regardless of the
    # measurement is worse than no conclusion.
    best = max(
        (f"planner (a_pen={p:g})" for p in ACTION_PENALTIES), key=lambda k: closed[k]
    )
    margin = closed[best]
    if margin < 5.0:
        print(
            f"\nThe planner does not beat doing nothing here ({margin:+.0f}% of the gap).\n"
            "Without an action penalty it moves a lot and ends further away -- exploiting\n"
            "model error. With a penalty it stops moving (see mean |a|) and merely\n"
            "reproduces the no-op baseline. The bottleneck is dynamics accuracy, not the\n"
            "cost or the planner: see the module docstring, and compare\n"
            "examples/05_planning.py where the same code works."
        )
    else:
        print(
            f"\n{best} closes {margin:+.0f}% of the gap to the goal, beating both the\n"
            "no-op and random baselines. The action penalty is what makes the difference:\n"
            "it is a measure of how far the latent dynamics can be trusted before the\n"
            "planner starts exploiting their error rather than the task. Compare mean |a|\n"
            "across the two penalties, and note that the oracle's 100% is free -- its\n"
            "action sequence is what generated the goal image."
        )

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    # Re-render the episode at PREVIEW_SIZE rather than upscaling the training
    # frames: nearest-neighbour upscaling only turns pixels into blocks.
    preview = []
    env.reset(seed=0)
    preview.append(env.render(PREVIEW_SIZE))
    for action in train["action"][0]:
        env.step(action)
        preview.append(env.render(PREVIEW_SIZE))
    report(xwm.plots.save_gif(out / "episode.gif", np.stack(preview), fps=5))
    fig = xwm.plots.plot_frames(
        train["video"][0],
        suptitle=f"Franka FR3 in Newton, {IMG_SIZE}x{IMG_SIZE} Warp raytraced observations",
    )
    report(xwm.plots.save_figure(fig, out / "episode_frames.png"))

    # A second look at the same two episodes, at figure quality. Both renders
    # are kept: the Warp strip above documents what the encoder actually sees,
    # this one shows what the robot is doing. Rendering must not cost the run
    # its other artifacts -- training already happened -- hence the guards.
    if not RTX:
        print("  (high-quality figures disabled: RTX=0)")
    else:
        backends = xwm.envs.which_backends()
        rendered = False
        if backends["rtx"]:
            try:
                hero = path_traced(env, train["action"][0], seed=0, size=RTX_SIZE)
                report(xwm.plots.save_gif(out / "episode_rtx.gif", hero, fps=5))
                fig = xwm.plots.plot_frames(
                    hero,
                    scale=RTX_SIZE / FIGURE_DPI,
                    suptitle=f"Franka FR3 in Newton, {RTX_SIZE}x{RTX_SIZE} OVRTX path traced",
                )
                report(
                    xwm.plots.save_figure(
                        fig, out / "episode_frames_rtx.png", dpi=FIGURE_DPI
                    )
                )

                if traces is not None:
                    order = ["planner", "oracle", "random"]
                    panels = [
                        path_traced(env, traces[k], seed=traces["seed"], size=RTX_SIZE)
                        for k in order
                    ]
                    report(
                        xwm.plots.save_gif(
                            out / "planning_episode_rtx.gif", panels, labels=order, fps=3
                        )
                    )
                rendered = True
            except Exception as exc:  # noqa: BLE001 - a figure is not worth the run
                print(f"  (OVRTX failed: {type(exc).__name__}: {exc})")

        if not rendered:
            # OVRTX needs a graphics-capable driver, which a compute-only GPU
            # container does not provide. Fall back to the best quality that is
            # available anywhere: the Warp raytracer at figure resolution with
            # supersampling, plus a USD stage so the same episode can be path
            # traced offline at whatever quality the renderer there allows.
            print("  (no OVRTX: writing supersampled Warp figures and a USD stage)")
            frames = []
            env.reset(seed=0)
            frames.append(env.render(RTX_SIZE, samples=RTX_SAMPLES))
            for action in train["action"][0]:
                env.step(action)
                frames.append(env.render(RTX_SIZE, samples=RTX_SAMPLES))
            frames = np.stack(frames)
            report(xwm.plots.save_gif(out / "episode_hq.gif", frames, fps=5))
            # One inch per FIGURE_DPI pixels, so each panel lands at the
            # resolution it was rendered at. The default strip is ~190 px per
            # panel, which would throw away everything the extra rays bought.
            fig = xwm.plots.plot_frames(
                frames,
                scale=RTX_SIZE / FIGURE_DPI,
                suptitle=(
                    f"Franka FR3 in Newton, {RTX_SIZE}x{RTX_SIZE} Warp raytraced, "
                    f"{RTX_SAMPLES}x supersampled"
                ),
            )
            report(xwm.plots.save_figure(fig, out / "episode_frames_hq.png", dpi=FIGURE_DPI))

            if xwm.envs.which_backends()["usd"]:
                with env.high_quality_renderer(
                    backend="usd", output_path=out / "episode.usd", fps=5
                ) as renderer:
                    env.reset(seed=0)
                    renderer.add(env.state)
                    for action in train["action"][0]:
                        env.step(action)
                        renderer.add(env.state)
                report(out / "episode.usd")

    if strips is not None:
        order = ["goal", "planner", "oracle", "random"]
        report(
            xwm.plots.save_gif(
                out / "planning_episode.gif",
                [strips[k] for k in order],
                labels=order,
                fps=3,
                scale=2,
            )
        )

    horizons = list(range(1, err.shape[0] + 1))
    ax = xwm.plots.plot_horizon(
        horizons,
        {"learned dynamics": [float(v) for v in err],
         "no-op baseline": [float(v) for v in base]},
        title="Franka: compounding error of a latent rollout",
    )
    report(xwm.plots.save_figure(ax.figure, out / "horizon_error.png"))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xwm.plots.plot_history(
        enc_history, keys=["loss_pred", "loss_reg"], logy=True, ax=axes[0],
        title="Stage 1: LeJEPA on frames",
    )
    xwm.plots.plot_history(
        dyn_history, keys=["loss_teacher_forcing", "loss_rollout"], logy=True, ax=axes[1],
        title="Stage 2: action-conditioned dynamics",
    )
    fig.tight_layout()
    report(xwm.plots.save_figure(fig, out / "training_curves.png"))

    # One pooled vector per episode's first frame, so the labels line up.
    ax = xwm.plots.plot_latent_pca(
        jnp.mean(z_test[:, 0], axis=-2),
        labels=jnp.asarray(test["joint_q"])[:, 0, 0],
        label_name="true joint 1 angle (rad)",
        title="Franka latents, coloured by a joint angle",
    )
    report(xwm.plots.save_figure(ax.figure, out / "latent_pca.png"))

    ax = xwm.plots.plot_bars(
        policies,
        [float(np.mean(plans[p])) for p in policies],
        ylabel="mean tool distance to goal (m)",
        title=f"Franka goal reaching, horizon {PLAN_HORIZON}, {PLAN_EPISODES} episodes",
    )
    report(xwm.plots.save_figure(ax.figure, out / "policy_comparison.png"))

    report(
        xwm.plots.save_table(
            out / "horizon_error",
            ["horizon", "model L1", "no-op L1", "ratio"],
            [[h, float(err[h - 1]), float(base[h - 1]), float(err[h - 1] / base[h - 1])]
             for h in horizons],
            caption="Franka: held-out latent rollout error against a no-op baseline.",
            label="tab:franka-horizon",
        )
    )
    report(
        xwm.plots.save_table(
            out / "planning",
            ["policy", "mean (m)", "median (m)", "worst (m)", "% of gap closed", "mean |a|"],
            [[p, float(np.mean(plans[p])), float(np.median(plans[p])),
              float(np.max(plans[p])), closed[p],
              float(np.mean(magnitudes[p])) if p in magnitudes else float("nan")]
             for p in policies],
            caption=(
                f"Franka tool distance to a goal image after {PLAN_HORIZON} steps. "
                "The planner optimises a latent cost and never sees these positions."
            ),
            label="tab:franka-planning",
        )
    )
    report(
        xwm.plots.save_table(
            out / "summary",
            ["quantity", "value"],
            [
                ["image size", IMG_SIZE],
                ["solver", env.solver_name],
                ["shadows", SHADOWS],
                ["textures", TEXTURES],
                ["training episodes", N_TRAIN],
                ["held-out episodes", N_TEST],
                ["frames per episode", CLIP_LEN],
                ["encoder loss (final)", enc_history[-1]["loss"]],
                ["teacher-forcing loss", dyn_history[-1]["loss_teacher_forcing"]],
                ["rollout loss", dyn_history[-1]["loss_rollout"]],
                ["joint-angle probe R2", float(probe["r2"])],
                ["planner % of gap closed", closed[f"planner (a_pen={ACTION_PENALTIES[0]:g})"]],
                ["rollout/no-op ratio at h=1", float(err[0] / base[0])],
                [f"rollout/no-op ratio at h={err.shape[0]}", float(err[-1] / base[-1])],
            ],
            caption="Franka world-model summary.",
            label="tab:franka-summary",
        )
    )


if __name__ == "__main__":
    main()
