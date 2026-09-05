"""Rerun logging: does a recording actually contain what the loggers claim?

Every assertion here reads the written ``.rrd`` back rather than trusting that
a log call returned without raising. A logger that silently writes to the wrong
entity path, or that writes one row where an episode had sixteen, is the failure
this file exists to catch -- and it is invisible to a smoke test that only
checks for exceptions.

Skipped entirely when the Rerun SDK is not installed.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

pytest.importorskip("rerun")

import xwm  # noqa: E402
from xwm.bench.control import EpisodeResult, StepInfo  # noqa: E402
from xwm.rerun import Recording  # noqa: E402


def read_back(path):
    """``entity path -> {"rows": int, "timelines": [...]}`` from a written recording.

    Rerun's own summary is a text block; parsing it here keeps the tests
    independent of whichever query API the SDK version ships.
    """
    from rerun.experimental import RrdReader

    summary = RrdReader(str(path)).store().summary()
    entities = {}
    for line in summary.splitlines():
        if not line.startswith("/"):
            continue
        entity, _, rest = line.partition(" ")
        parts = [p for p in rest.split(" ") if "=" in p and not p.startswith("cols")]
        fields = dict(part.split("=", 1) for part in parts)
        # One line per *chunk*, and a long stream is split into several, so the
        # counts accumulate. Reading only the last line would under-report every
        # entity logged across more than one flush.
        seen = entities.setdefault(entity, {"rows": 0, "timelines": False})
        seen["rows"] += int(fields.get("rows", 0))
        seen["timelines"] = seen["timelines"] or "timelines=[]" not in rest
    return entities


@pytest.fixture
def recording(tmp_path):
    """A recording writing to a file, closed at the end of the test."""
    path = tmp_path / "test.rrd"
    rec = Recording("xwm-test", path=path)
    yield rec, path
    rec.close()


def _tiny_model(key):
    from conftest import TINY_ENCODER, TINY_PREDICTOR

    return xwm.families.jepa.ijepa(
        img_size=16,
        patch_size=8,
        encoder_kwargs=TINY_ENCODER,
        predictor_kwargs=TINY_PREDICTOR,
        key=key,
    )


# -- training ---------------------------------------------------------------
def test_training_callback_logs_every_metric_on_the_step_timeline(tmp_path, key):
    """The callback is the whole training integration; it must log what fit reports."""
    path = tmp_path / "train.rrd"
    model = _tiny_model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
    batch = {"image": jr.normal(key, (2, 3, 16, 16))}
    batches = (dict(batch) for _ in range(10))

    with Recording("xwm-test", path=path) as rec:
        state, history = trainer.fit(
            batches, key=key, steps=10, log_every=2, callbacks=[xwm.rerun.training_callback(rec)]
        )

    entities = read_back(path)
    assert "/train/loss" in entities, sorted(entities)
    # fit calls its callbacks on the steps it appends to history, and no others.
    assert entities["/train/loss"]["rows"] == len(history)
    assert entities["/train/loss"]["timelines"]


def test_history_is_logged_columnwise_with_one_row_per_entry(tmp_path):
    """A finished history goes in one send per metric, not one per step."""
    path = tmp_path / "history.rrd"
    history = [{"step": i * 5, "loss": 1.0 / (i + 1), "grad_norm": 0.5} for i in range(20)]
    with Recording("xwm-test", path=path) as rec:
        xwm.rerun.log_history(rec, history)

    entities = read_back(path)
    assert entities["/train/loss"]["rows"] == 20
    assert entities["/train/grad_norm"]["rows"] == 20


def test_collapse_report_and_its_spectrum_are_both_logged(tmp_path, key):
    """The loss is not the metric: the diagnostics must reach the same timeline."""
    path = tmp_path / "collapse.rrd"
    z = jr.normal(key, (64, 16))
    with Recording("xwm-test", path=path) as rec:
        xwm.rerun.log_collapse(rec, z, step=0)

    entities = read_back(path)
    for name in ("rankme", "rank_ratio", "feature_std", "mean_cosine"):
        assert f"/collapse/{name}" in entities
    assert "/collapse/spectrum" in entities


# -- latents and plans ------------------------------------------------------
def test_latent_trajectory_logs_both_paths_and_a_per_step_error(tmp_path, key):
    """The imagined path, the real one, and the distance the projection hides."""
    path = tmp_path / "latent.rrd"
    horizon = 6
    true_z = jr.normal(key, (horizon, 4, 8))
    imagined_z = true_z + 0.1 * jr.normal(jr.fold_in(key, 1), (horizon, 4, 8))

    with Recording("xwm-test", path=path) as rec:
        xwm.rerun.log_latent_trajectory(rec, true_z, imagined_z)

    entities = read_back(path)
    assert "/latent/true" in entities and "/latent/imagined" in entities
    assert entities["/latent/error"]["rows"] == horizon


def test_latent_trajectory_refuses_mismatched_shapes(key):
    """A silent shape mismatch would plot two unrelated paths as if comparable."""
    with pytest.raises(ValueError, match="differ"):
        xwm.rerun.log_latent_trajectory(
            Recording("xwm-test"), jr.normal(key, (4, 8)), jr.normal(key, (5, 8))
        )


def test_plan_logs_its_spread_not_only_the_action_taken(tmp_path, key):
    """A plan whose std never shrinks is one the search never converged on."""
    path = tmp_path / "plan.rrd"
    planner = xwm.planning.CEM(horizon=4, action_dim=2, n_samples=32, n_elites=8, n_iters=2)
    plan = planner.plan(
        key,
        lambda z, a: z + jnp.pad(a, (0, z.shape[-1] - a.shape[-1])),
        jnp.zeros((6,)),
        xwm.planning.goal_cost(jnp.ones((6,))),
    )
    with Recording("xwm-test", path=path) as rec:
        xwm.rerun.log_plan(rec, plan)

    entities = read_back(path)
    assert entities["/plan/mean/0"]["rows"] == 4
    assert entities["/plan/std/0"]["rows"] == 4
    assert "/plan/cost" in entities


# -- benchmark --------------------------------------------------------------
def _episode_result(steps=8):
    rng = np.random.default_rng(0)
    return EpisodeResult(
        success=True,
        steps=steps,
        initial_distance=0.5,
        final_distance=0.1,
        best_distance=0.1,
        solved_at=steps - 1,
        distances=[0.5 - 0.05 * i for i in range(steps + 1)],
        frames=[rng.random((3, 8, 8), dtype=np.float32) for _ in range(steps + 1)],
    )


def test_finished_episode_logs_its_distance_trace_and_frames(tmp_path):
    """A results directory from last week must be replayable without re-running it."""
    path = tmp_path / "episode.rrd"
    result = _episode_result()
    with Recording("xwm-test", path=path) as rec:
        xwm.rerun.log_episode(rec, result, name="planner")

    entities = read_back(path)
    assert entities["/episode/planner/distance"]["rows"] == len(result.distances)
    assert entities["/episode/planner/observation"]["rows"] == len(result.frames)
    assert "/episode/planner/outcome" in entities


def test_episode_hook_logs_one_row_per_raw_environment_step(tmp_path):
    """The budget is counted in raw steps, so the log must be too."""
    path = tmp_path / "hook.rrd"
    with Recording("xwm-test", path=path) as rec:
        hook = xwm.rerun.episode_hook(rec)
        for step in range(1, 13):
            hook(
                "planner",
                0,
                StepInfo(
                    step=step,
                    action=np.zeros(4, np.float32),
                    raw=np.zeros(2, np.float32),
                    state=np.zeros(6, np.float32),
                    goal_state=np.zeros(6, np.float32),
                    distance=1.0 / step,
                    success=False,
                ),
            )

    entities = read_back(path)
    assert entities["/episode/planner/distance"]["rows"] == 12
    assert "/episode/planner/action/0" in entities


def test_hooks_and_loggers_are_no_ops_without_a_recording(key):
    """`None` must be a valid recording, or every call site grows a branch."""
    assert xwm.rerun.episode_hook(None) is None
    xwm.rerun.log_collapse(None, jr.normal(key, (8, 4)), step=0)
    xwm.rerun.log_episode(None, _episode_result())
    xwm.rerun.log_history(None, [{"step": 0, "loss": 1.0}])
    callback = xwm.rerun.training_callback(None)
    callback(object(), {"loss": 1.0})


# -- the optional dependency ------------------------------------------------
def test_the_helper_names_the_extra_when_the_sdk_is_missing(monkeypatch):
    """Importing xwm must work without the SDK; calling a helper must say why not."""
    import builtins

    from xwm.rerun import _backend

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "rerun" or name.startswith("rerun."):
            raise ImportError("no module named rerun")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(ImportError, match=r"xwm\[rerun\]"):
        _backend.rerun()


# -- the Franka scene -------------------------------------------------------
def test_franka_scene_logs_articulated_geometry_and_a_camera(tmp_path):
    """The 3-D view must contain a robot, not an empty frame tree.

    The failure this catches is silent: Rerun logs a URDF whose ``package://``
    meshes it cannot resolve without raising, and the viewer opens on nothing.
    """
    pytest.importorskip("newton")
    pytest.importorskip("warp")

    path = tmp_path / "franka.rrd"
    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=32))
    with Recording("xwm-test", path=path) as rec:
        hook = xwm.rerun.franka_hook(rec, env)
        env.reset(seed=0)
        for _ in range(3):
            env.step(np.full(env.action_dim, 0.2, np.float32))
            hook()

    entities = read_back(path)
    meshes = [e for e in entities if "visual_geometries" in e]
    assert meshes, "no visual geometry: the URDF's meshes did not resolve"
    # One transform per movable joint per step, all on one entity.
    assert entities["/franka/joints"]["rows"] == 3 * env.action_dim
    assert entities["/franka/camera"]["rows"] >= 1
    assert entities["/franka/signal/goal_distance"]["rows"] == 3
    assert entities["/franka/camera/image"]["rows"] == 3


def test_resolved_urdf_leaves_newtons_cache_alone(tmp_path):
    """A rewritten URDF is a copy: another library's cache is not ours to edit."""
    pytest.importorskip("newton")
    from xwm.rerun.franka import resolve_urdf

    env = xwm.envs.FrankaEnv(xwm.envs.FrankaConfig(image_size=32))
    original = env.urdf_path
    before = original.read_text()
    resolved = resolve_urdf(original)

    assert resolved != original
    assert original.read_text() == before
    assert "package://" not in resolved.read_text()
