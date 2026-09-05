"""Windowed world models: LeWorldModel, Delta-JEPA and DINO-WM.

These are the first models in the library whose planning state is more than one
frame, so most of what is tested here is the *window*: that a future frame
cannot leak backwards through the predictor, that the planner's state slides
correctly, and that the goal cost is compared against the newest frame rather
than the stale context.

The other theme is collapse. Both pooled recipes train the encoder through its
own targets, which is the configuration that admits the constant solution, so
each one's countermeasure is tested by removing it and watching the failure.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import TINY_ENCODER

import xwm
from xwm.dynamics import ARPredictor, ContinuousActionEmbed, block_causal_mask
from xwm.families.jepa.autoregressive import ARWorldModel, Window

TINY_DYN = {"depth": 2, "pred_dim": 32, "num_heads": 4}


def _lewm(key, **kw):
    return xwm.families.jepa.lewm(
        key=key, action_dim=2, img_size=16, patch_size=8,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, n_proj=64, **kw,
    )


def _delta(key, **kw):
    return xwm.families.jepa.delta_jepa(
        key=key, action_dim=2, img_size=16, patch_size=8,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, **kw,
    )


def _dinowm(key, history=3, **kw):
    encoder = xwm.encoders.DINOv2Encoder(
        img_size=28, embed_dim=64, depth=2, num_heads=4, key=key
    )
    return xwm.families.dinowm.dinowm(
        key=key, action_dim=2, history=history, encoder=encoder, img_size=28,
        dynamics_kwargs=TINY_DYN, **kw,
    )


def _batch(key, *, history=3, batch=4, size=16):
    return {
        "video": jr.normal(key, (batch, history + 1, 3, size, size)),
        "action": jr.normal(jr.fold_in(key, 1), (batch, history, 2)),
    }


def _grad_norm(tree):
    leaves = [
        jnp.sum(x**2) for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    ]
    return float(jnp.sqrt(sum(leaves))) if leaves else 0.0


# -- the predictor ----------------------------------------------------------
def test_a_frame_cannot_see_its_own_future(key):
    """The whole point of the causal mask; without it the task is trivial."""
    embed = ContinuousActionEmbed(2, 32, key=key)
    predictor = ARPredictor((2, 2), 32, embed, history=3, key=key, **TINY_DYN)
    z, actions = jr.normal(key, (3, 4, 32)), jr.normal(jr.fold_in(key, 1), (3, 2))
    baseline = predictor(z, actions)

    moved = predictor(z.at[-1].add(3.0), actions)
    assert jnp.allclose(baseline[:-1], moved[:-1], atol=1e-5), "a later frame moved an earlier one"

    later_action = predictor(z, actions.at[-1].add(3.0))
    assert jnp.allclose(baseline[:-1], later_action[:-1], atol=1e-5)
    assert not jnp.allclose(baseline[-1], later_action[-1], atol=1e-4), "the action did nothing"


def test_tokens_of_one_frame_see_each_other(key):
    """Block causal, not causal: patches of a single image have no natural order."""
    mask = block_causal_mask(3, 4)
    assert mask.shape == (12, 12)
    assert bool(mask[0, 3]) and bool(mask[3, 0]), "tokens within a frame must attend both ways"
    assert not bool(mask[0, 4]), "frame 0 must not see frame 1"
    assert bool(mask[4, 0]), "frame 1 must see frame 0"


@pytest.mark.parametrize("conditioning", ["film", "concat", "both"])
def test_every_conditioning_route_reaches_the_prediction(conditioning, key):
    """A dynamics model that ignores its action is the failure that looks like underfitting."""
    embed = ContinuousActionEmbed(2, 32, key=key)
    predictor = ARPredictor(
        (2, 2), 32, embed, history=2, key=key, conditioning=conditioning, **TINY_DYN
    )
    z = jr.normal(key, (2, 4, 32))
    a = jnp.zeros((2, 2))
    spread = jnp.abs(predictor(z, a.at[-1].set(1.0)) - predictor(z, a.at[-1].set(-1.0))).max()
    assert float(spread) > 1e-4


def test_the_predictor_rejects_a_window_of_the_wrong_length(key):
    """Silently accepting it would mean a temporal embedding read at the wrong offset."""
    embed = ContinuousActionEmbed(2, 32, key=key)
    predictor = ARPredictor((2, 2), 32, embed, history=3, key=key, **TINY_DYN)
    with pytest.raises(ValueError, match="expected 3 frames"):
        predictor(jr.normal(key, (2, 4, 32)), jr.normal(key, (2, 2)))


# -- the models -------------------------------------------------------------
@pytest.mark.parametrize("build", [_lewm, _delta, _dinowm], ids=["lewm", "delta", "dinowm"])
def test_loss_is_finite_and_reports_its_terms(build, key):
    model = build(key)
    size = 28 if build is _dinowm else 16
    loss, metrics = model.loss(_batch(key, size=size), key=key)
    assert jnp.isfinite(loss)
    assert "loss_prediction" in metrics and "latent_std" in metrics
    assert float(metrics["latent_std"]) > 0.01, "the latent is already degenerate at init"


@pytest.mark.parametrize("build", [_lewm, _delta, _dinowm], ids=["lewm", "delta", "dinowm"])
def test_gradients_reach_every_component_that_should_move(build, key):
    """And none of the ones that should not."""
    model = build(key)
    size = 28 if build is _dinowm else 16
    batch = _batch(key, size=size)
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)

    assert _grad_norm(grads.dynamics) > 0, "no gradient reached the dynamics"
    if model.freeze_encoder:
        assert _grad_norm(grads.encoder) == 0, "a frozen encoder received gradient"
    else:
        assert _grad_norm(grads.encoder) > 0, "the encoder is trained end to end but got nothing"
    if model.action_decoder is not None:
        assert _grad_norm(grads.action_decoder) > 0


def test_a_frozen_encoder_costs_no_optimizer_state(key):
    """The saving is the point of freezing, not just the fixed targets."""
    model = _dinowm(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-4))
    assert trainer.n_trainable == model.dynamics.n_params
    assert trainer.n_trainable < model.n_params


@pytest.mark.parametrize("build", [_lewm, _delta], ids=["lewm", "delta"])
def test_the_loss_moves_under_the_family_agnostic_trainer(build, key):
    model = build(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    batches = (_batch(jr.fold_in(key, i)) for i in range(12))
    state, history = trainer.fit(batches, key=key, steps=12, log_every=1)
    assert int(state.step) == 12
    assert state.target is None and model.uses_target is False
    assert history[-1]["loss"] != history[0]["loss"]


def test_a_clip_of_the_wrong_length_is_refused_by_name(key):
    """The window and the clip have to agree, and the message must say which to change."""
    model = _lewm(key)
    with pytest.raises(ValueError, match="conditions on 3 frames"):
        model.loss(_batch(key, history=2), key=key)


def test_an_unguarded_end_to_end_model_is_refused_at_construction(key):
    """A live encoder, movable targets and no countermeasure trains a constant."""
    base = _lewm(key)
    with pytest.raises(ValueError, match="will collapse"):
        ARWorldModel(base.encoder, base.dynamics, projector=base.projector, collapse="none")


def test_delta_jepa_needs_a_decoder_for_its_action_term(key):
    base = _lewm(key)
    with pytest.raises(ValueError, match="needs an action_decoder"):
        ARWorldModel(
            base.encoder, base.dynamics, projector=base.projector,
            collapse="none", action_weight=1.0,
        )


def test_the_action_decoder_learns_to_read_the_action_off_the_displacement(key):
    """Delta-JEPA's whole mechanism: no recoverable action, no protection."""
    decoder = xwm.heads.LatentDifferenceActionDecoder(8, 2, key=key)
    deltas = jr.normal(key, (64, 8))
    actions = deltas[:, :2] * 2.0
    trained = decoder

    @eqx.filter_jit
    def step(model, opt_state, optimizer):
        def loss(m):
            return jnp.mean(jax.vmap(m.loss)(deltas, actions))

        value, grads = eqx.filter_value_and_grad(loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, value

    optimizer = xwm.training.adamw(1e-2)
    opt_state = optimizer.init(eqx.filter(trained, eqx.is_array))
    first = None
    for _ in range(200):
        trained, opt_state, value = step(trained, opt_state, optimizer)
        first = value if first is None else first
    assert float(value) < 0.5 * float(first), "the decoder did not learn a linear map"


# -- planning ---------------------------------------------------------------
@pytest.mark.parametrize("build", [_lewm, _delta, _dinowm], ids=["lewm", "delta", "dinowm"])
def test_the_window_satisfies_the_planning_contract(build, key):
    """`xwm.bench` talks to a model through this and nothing else."""
    model = build(key)
    assert isinstance(model, xwm.core.Plannable)
    assert model.history == 3

    size = 28 if build is _dinowm else 16
    frames = jr.normal(key, (3, 3, size, size))
    window = model.initial_state(frames, jr.normal(key, (2, 2)))
    assert isinstance(window, Window)
    assert window.frames.shape == (3, *model.latent_shape)
    assert window.actions.shape == (3, 2)
    assert jnp.all(window.actions[-1] == 0), "the newest action slot is the planner's to fill"
    assert model.readout(window).shape == model.latent_shape
    assert model.goal_embedding(frames[0]).shape == model.latent_shape


def test_stepping_the_window_slides_it(key):
    """The oldest frame falls off, the prediction joins, the action lands in its slot."""
    model = _lewm(key)
    frames = jr.normal(key, (3, 3, 16, 16))
    past = jr.normal(jr.fold_in(key, 2), (2, 2))
    window = model.initial_state(frames, past)
    action = jnp.array([0.5, -0.5])
    stepped = model.dynamics_fn()(window, action)

    assert jnp.allclose(stepped.frames[:-1], window.frames[1:]), "the window did not slide"
    assert not jnp.allclose(stepped.frames[-1], window.frames[-1]), "nothing was predicted"
    assert jnp.allclose(stepped.actions[-2], action), "the action was not recorded in the window"
    assert jnp.all(stepped.actions[-1] == 0), "the new slot was not cleared"
    assert jnp.allclose(stepped.actions[0], past[-1])


def test_initial_state_keeps_the_most_recent_actions(key):
    """Given more actions than the window holds, the *newest* are the ones that matter."""
    model = _lewm(key)
    frames = jr.normal(key, (3, 3, 16, 16))
    actions = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    window = model.initial_state(frames, actions)
    assert jnp.allclose(window.actions[:2], actions[-2:])


def test_a_planner_reaches_a_goal_through_the_windowed_dynamics(key):
    """initial_state -> dynamics_fn -> readout, composed by a real CEM plan."""
    model = _lewm(key)
    frames = jr.normal(key, (3, 3, 16, 16))
    window = model.initial_state(frames, jr.normal(key, (2, 2)))
    goal = model.goal_embedding(jr.normal(jr.fold_in(key, 5), (3, 16, 16)))
    cost = xwm.planning.goal_cost(goal, kind="l2", readout=model.readout)
    planner = xwm.planning.CEM(horizon=4, action_dim=2, n_samples=64, n_elites=8, n_iters=3)

    plan = eqx.filter_jit(planner.plan)(key, model.dynamics_fn(), window, cost)
    assert plan.actions.shape == (4, 2)
    assert jnp.isfinite(plan.cost)

    # The plan must beat doing nothing, or the search is not using the model.
    def score(actions):
        return sum(
            float(cost(w, a, t))
            for t, (w, a) in enumerate(
                zip(_unroll(model, window, actions), actions, strict=True)
            )
        )

    assert score(plan.actions) < score(jnp.zeros_like(plan.actions))


def _unroll(model, window, actions):
    step = model.dynamics_fn()
    out = []
    for action in actions:
        window = step(window, action)
        out.append(window)
    return out


def test_imagine_returns_a_stacked_window_per_step(key):
    model = _lewm(key)
    window = model.initial_state(jr.normal(key, (3, 3, 16, 16)), jr.normal(key, (2, 2)))
    traj = model.imagine(window, jr.normal(key, (5, 2)))
    assert traj.frames.shape == (5, 3, *model.latent_shape)


def test_history_of_one_is_still_a_valid_window(key):
    """The degenerate case has to work, or the Markov tasks cannot use these models."""
    model = _lewm(key, history=1)
    window = model.initial_state(jr.normal(key, (1, 3, 16, 16)), None)
    assert window.frames.shape == (1, *model.latent_shape)
    stepped = model.dynamics_fn()(window, jnp.ones((2,)))
    assert stepped.frames.shape == window.frames.shape


# -- registry and config ----------------------------------------------------
def test_the_new_families_are_registered_and_buildable_by_name():
    assert {"jepa/lewm", "jepa/delta", "dinowm"} <= set(xwm.families.available())
    assert "dinowm" in xwm.families.families()
    model = xwm.families.create(
        "jepa/lewm", action_dim=2, history=2, img_size=16, patch_size=8,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN,
    )
    assert isinstance(model, ARWorldModel)
    assert model.history == 2


def test_lewm_does_not_collapse_on_a_real_world(key):
    """The failure that shipped once: SIGReg present, latent collapsed anyway.

    An earlier version normalised each pooled embedding onto the unit sphere
    before predicting on it. That removes the scale SIGReg constrains, so the
    prediction loss could be driven to zero by mapping every frame to a single
    point while the regularizer sat at the value it takes on a point mass.

    Scope: this pins the *layer norm*, on this world, at the default weight. It
    is not a guarantee that ``reg_weight`` never needs raising -- on the
    benchmark task it does, and `docs/findings.md` records by how much. Trained
    on a real world rather than noise, because collapse needs structure to
    collapse away from.
    """
    world = xwm.data.PushWorld(32)
    data = xwm.data.push_sequences(jr.fold_in(key, 1), 128, 4, world=world)
    model = xwm.families.jepa.lewm(
        key=key, action_dim=2, history=3, img_size=32, patch_size=8,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, n_proj=128,
    )
    trainer = xwm.training.Trainer(model, xwm.training.adamw(3e-4))
    batches = xwm.data.iter_batches(
        {"video": data["video"], "action": data["action"]}, 32, key=key, epochs=None
    )
    _, history = trainer.fit(batches, key=key, steps=300, log_every=50)

    assert history[-1]["latent_std"] > 0.1, (
        f"latent collapsed: latent_std {history[0]['latent_std']:.4f} -> "
        f"{history[-1]['latent_std']:.4f} at the paper's own reg_weight"
    )
    # SIGReg's own value is the second witness: a point mass scores ~0.43 on
    # this statistic and an isotropic batch scores ~0.00.
    assert history[-1]["loss_reg"] < 0.2


def test_a_projected_latent_reports_its_own_width(key):
    """`latent_dim` moves the whole latent space, so every shape must follow it."""
    model = xwm.families.jepa.lewm(
        key=key, action_dim=2, history=2, img_size=16, patch_size=8, latent_dim=32,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, n_proj=64,
    )
    assert model.encoder.embed_dim == TINY_ENCODER["embed_dim"] != 32
    assert model.latent_shape == (1, 32)
    assert model.encode(jr.normal(key, (3, 16, 16))).shape == (1, 32)

    window = model.initial_state(jr.normal(key, (2, 3, 16, 16)), jr.normal(key, (1, 2)))
    assert window.frames.shape == (2, 1, 32)
    assert model.readout(window).shape == (1, 32)


def test_a_dynamics_of_the_wrong_width_is_refused(key):
    """Silently accepting it would fail much later, inside a vmapped matmul."""
    narrow = xwm.families.jepa.lewm(
        key=key, action_dim=2, history=2, img_size=16, patch_size=8, latent_dim=32,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, n_proj=64,
    )
    wide = xwm.families.jepa.lewm(
        key=key, action_dim=2, history=2, img_size=16, patch_size=8,
        encoder_kwargs=TINY_ENCODER, dynamics_kwargs=TINY_DYN, n_proj=64,
    )
    with pytest.raises(ValueError, match="width"):
        ARWorldModel(wide.encoder, narrow.dynamics, collapse="sigreg")
