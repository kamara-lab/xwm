"""World models: forward passes, gradient routing, jit-stability, freezing."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import xwm

ENC = {"depth": 2, "embed_dim": 64, "num_heads": 4}
PRED = {"depth": 2, "pred_dim": 32, "num_heads": 4}


def _grad_norm(tree):
    leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))) if leaves else 0.0


# -- backbones --------------------------------------------------------------
def test_encoder_output_depends_on_which_tokens_are_kept(key):
    """The masked path must be position-aware, not just count-aware.

    Encoding a subset is deliberately *not* the same as encoding everything and
    slicing -- attention mixes tokens, so the context encoder sees a genuinely
    different problem. What must hold is that the result depends on *which*
    indices were kept, not merely how many.
    """
    enc = xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC).eval_mode()
    x = jr.normal(key, (3, 16, 16))
    a = enc(x, keep=jnp.array([0, 1, 2]))
    b = enc(x, keep=jnp.array([5, 9, 13]))
    assert a.shape == b.shape == (3, 64)
    assert not jnp.allclose(a, b)


def test_predictor_output_depends_on_target_position(key):
    """A predictor that ignores target positions is not solving the JEPA task."""
    enc = xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC).eval_mode()
    pred = xwm.families.jepa.JEPAPredictor(enc.grid, enc.embed_dim, key=key, **PRED).eval_mode()
    ctx_idx = jnp.array([0, 1, 2, 3])
    ctx = enc(jr.normal(key, (3, 16, 16)), keep=ctx_idx)
    a = pred(ctx, ctx_idx, jnp.array([8, 9]))
    b = pred(ctx, ctx_idx, jnp.array([14, 15]))
    assert a.shape == b.shape == (2, 64)
    assert not jnp.allclose(a, b)


def test_jepa_rejects_mismatched_parts(key):
    enc = xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC)
    sampler = xwm.masking.MultiBlockMask2d(enc.grid, n_targets=2)
    wrong_width = xwm.families.jepa.JEPAPredictor(enc.grid, 128, key=key, **PRED)
    with pytest.raises(ValueError, match="width"):
        xwm.JEPA(enc, wrong_width, sampler)
    wrong_grid = xwm.families.jepa.JEPAPredictor((2, 2), enc.embed_dim, key=key, **PRED)
    with pytest.raises(ValueError, match="grid mismatch"):
        xwm.JEPA(enc, wrong_grid, sampler)


# -- JEPA -------------------------------------------------------------------
def _image_model(key, **kw):
    return xwm.families.jepa.ijepa(
        key=key, img_size=16, patch_size=4, n_targets=2, encoder_kwargs=ENC,
        predictor_kwargs=PRED, **kw
    )


@pytest.mark.parametrize("collapse", ["ema", "sigreg", "vicreg", "none"])
def test_all_collapse_strategies_train(collapse, key):
    base = _image_model(key)
    model = xwm.JEPA(
        base.encoder, base.predictor, base.mask_sampler, input_key="image",
        collapse=collapse, reg_weight=0.1,
    )
    assert model.uses_target == (collapse == "ema")
    batch = model.prepare_batch({"image": jr.normal(key, (3, 3, 16, 16))}, key)
    target = model if model.uses_target else None
    loss, metrics = model.loss(batch, key=key, target=target)
    assert jnp.ndim(loss) == 0 and bool(jnp.isfinite(loss))
    assert {"loss", "loss_pred", "loss_reg", "embed_std"} <= set(metrics)
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key, target=target)[0])(model)
    assert _grad_norm(grads) > 0
    if collapse in ("sigreg", "vicreg"):
        assert float(metrics["loss_reg"]) > 0
    else:
        assert float(metrics["loss_reg"]) == 0


def test_jepa_requires_masks(key):
    model = _image_model(key)
    with pytest.raises(KeyError, match="prepare_batch"):
        model.loss({"image": jr.normal(key, (2, 3, 16, 16))}, key=key, target=model)


def test_ema_target_receives_no_gradient(key):
    """Gradients must never flow into the teacher branch."""
    model = _image_model(key)
    batch = model.prepare_batch({"image": jr.normal(key, (3, 3, 16, 16))}, key)
    target = jax.tree_util.tree_map(lambda x: x, model)
    grads = eqx.filter_grad(
        lambda t: model.loss(batch, key=key, target=t)[0]
    )(target)
    assert _grad_norm(grads) == 0.0


def test_lejepa_flows_gradient_through_both_branches(key):
    """LeJEPA's whole point: no stop-gradient, so the target branch trains too."""
    ema = _image_model(key)
    lejepa = xwm.families.jepa.lejepa(
        key=key, img_size=16, patch_size=4, n_targets=2, encoder_kwargs=ENC,
        predictor_kwargs=PRED, reg_weight=1.0,
    )
    batch = lejepa.prepare_batch({"image": jr.normal(key, (3, 3, 16, 16))}, key)
    g_lejepa = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(lejepa)
    g_ema = eqx.filter_grad(lambda m: m.loss(batch, key=key, target=ema)[0])(ema)
    # Both train the encoder, but LeJEPA has no teacher to maintain at all.
    assert _grad_norm(g_lejepa.encoder) > 0
    assert _grad_norm(g_ema.encoder) > 0
    assert ema.uses_target and not lejepa.uses_target


def test_video_model_shapes(key):
    model = xwm.families.jepa.vjepa(
        key=key, img_size=16, patch_size=4, num_frames=4, tubelet_size=2,
        n_targets=2, spatial_scale=0.2, encoder_kwargs=ENC, predictor_kwargs=PRED,
    )
    # A 3-axis grid with an ordinary width: the axis split must handle 64 / 3.
    assert model.encoder.grid == (2, 4, 4)
    batch = model.prepare_batch({"video": jr.normal(key, (2, 4, 3, 16, 16))}, key)
    loss, _ = model.loss(batch, key=key, target=model)
    assert bool(jnp.isfinite(loss))


def test_step_is_traced_once_despite_fresh_masks(key):
    """Masks are resampled every step; static shapes must keep the trace count at 1."""
    model = _image_model(key)
    traces = {"n": 0}

    @eqx.filter_jit
    def step(m, batch, k):
        traces["n"] += 1
        return m.loss(batch, key=k, target=m)[0]

    for i in range(5):
        k = jr.fold_in(key, i)
        batch = model.prepare_batch({"image": jr.normal(k, (2, 3, 16, 16))}, k)
        step(model, batch, key)
    assert traces["n"] == 1


# -- action-conditioned ------------------------------------------------------
def _action_model(key, **kw):
    return xwm.families.jepa.action_world_model(
        key=key, action_dim=2, img_size=16, patch_size=4,
        encoder=xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC),
        dynamics_kwargs=PRED, **kw
    )


def test_action_model_loss_and_metrics(key):
    model = _action_model(key)
    batch = {"video": jr.normal(key, (2, 5, 3, 16, 16)), "action": jr.normal(key, (2, 4, 2))}
    loss, metrics = model.loss(batch, key=key)
    assert bool(jnp.isfinite(loss))
    assert {"loss_teacher_forcing", "loss_rollout", "latent_std"} <= set(metrics)
    # Compounding error: a free rollout should be no easier than teacher forcing.
    assert float(metrics["loss_rollout"]) >= float(metrics["loss_teacher_forcing"])


def test_action_model_rejects_mismatched_action_count(key):
    model = _action_model(key)
    with pytest.raises(ValueError, match="expected 4 actions"):
        model.loss(
            {"video": jr.normal(key, (2, 5, 3, 16, 16)), "action": jr.normal(key, (2, 9, 2))},
            key=key,
        )


def test_detached_target_starves_the_encoder_of_half_its_gradient(key):
    """``detach_target`` decides whether gradient flows through the target branch.

    Detaching is right for a frozen encoder and for the V-JEPA 2-AC recipe; the
    end-to-end LeJEPA-family models train *through* the target and rely on a
    regularizer to stop the encoder trivialising it. Both must be expressible,
    and the difference has to be visible in the gradient.
    """
    batch = {"video": jr.normal(key, (2, 4, 3, 16, 16)), "action": jr.normal(key, (2, 3, 2))}
    grad_norm = lambda m: _grad_norm(  # noqa: E731
        eqx.filter_grad(lambda x: x.loss(batch, key=key)[0])(m).encoder
    )
    detached = grad_norm(_action_model(key, freeze_encoder=False, detach_target=True))
    live = grad_norm(_action_model(key, freeze_encoder=False, detach_target=False))
    assert detached > 0
    assert live > detached


def test_detached_target_is_the_default(key):
    assert _action_model(key).detach_target is True


def test_frozen_encoder_gets_no_gradient_and_no_optimizer_state(key):
    model = _action_model(key, freeze_encoder=True)
    batch = {"video": jr.normal(key, (2, 4, 3, 16, 16)), "action": jr.normal(key, (2, 3, 2))}
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)
    assert _grad_norm(grads.encoder) == 0.0
    assert _grad_norm(grads.dynamics) > 0
    assert not any(jax.tree_util.tree_leaves(model.trainable().encoder))
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    assert trainer.n_trainable == model.dynamics.n_params


def test_unfrozen_encoder_is_trainable(key):
    model = _action_model(key, freeze_encoder=False)
    batch = {"video": jr.normal(key, (2, 4, 3, 16, 16)), "action": jr.normal(key, (2, 3, 2))}
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)
    assert _grad_norm(grads.encoder) > 0
    assert any(jax.tree_util.tree_leaves(model.trainable().encoder))


@pytest.mark.parametrize("conditioning", ["token", "film", "both"])
def test_action_pathway_is_live_at_init(conditioning, key):
    """Gradients must reach the action embedding from step one.

    This is what a fully zero-initialised output projection would break: it
    zeroes the gradient flowing back through it, stalling the whole trunk.
    """
    model = _action_model(key, conditioning=conditioning)
    batch = {"video": jr.normal(key, (2, 4, 3, 16, 16)), "action": jr.normal(key, (2, 3, 2))}
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)
    assert _grad_norm(grads.dynamics.action_embed) > 0
    assert _grad_norm(grads.dynamics.blocks) > 0
    assert _grad_norm(grads.dynamics.embed_out) > 0


@pytest.mark.parametrize("conditioning", ["token", "film", "both"])
def test_dynamics_learns_to_respond_to_the_action(conditioning, key):
    """After training, the dynamics must distinguish different actions.

    A model that ignores its action input is useless for planning, and because
    the residual head is zero-initialised this only becomes checkable after a
    few optimizer steps.
    """
    world = xwm.data.SpriteWorld(16, n_distractors=0)
    data = xwm.data.sprite_sequences(key, 128, 5, world=world)
    model = _action_model(key, conditioning=conditioning)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(3e-3))
    # 150 steps: "token" conditioning needs attention to learn to read the
    # action token, so it is measurably slower off the mark than FiLM.
    state, _ = trainer.fit(
        xwm.data.iter_batches(
            {"video": data["video"], "action": data["action"]}, 8, key=key, epochs=None
        ),
        steps=150, key=key, log_every=0,
    )
    dynamics = state.model.dynamics.eval_mode()
    z = state.model.encode(data["video"][0, 0])
    left = dynamics(z, jnp.array([-1.0, 0.0]))
    right = dynamics(z, jnp.array([1.0, 0.0]))
    spread = float(jnp.linalg.norm(left - right)) / float(jnp.linalg.norm(z))
    assert spread > 5e-3, f"dynamics barely distinguishes opposite actions ({spread:.2e})"


def test_residual_dynamics_starts_near_the_identity(key):
    """A fresh residual model must barely perturb the latent, but not be frozen."""
    model = _action_model(key)
    z = model.encode(jr.normal(key, (3, 16, 16)))
    drift = float(jnp.linalg.norm(model.dynamics.eval_mode()(z, jnp.zeros((2,))) - z))
    relative = drift / float(jnp.linalg.norm(z))
    assert 0.0 < relative < 0.1, f"initial drift {relative:.3f} is not near-identity"


def test_residual_head_still_receives_gradient(key):
    """Zero-init must not mean zero gradient -- the layer's input is non-zero."""
    model = _action_model(key)
    batch = {"video": jr.normal(key, (2, 4, 3, 16, 16)), "action": jr.normal(key, (2, 3, 2))}
    grads = eqx.filter_grad(lambda m: m.loss(batch, key=key)[0])(model)
    assert _grad_norm(grads.dynamics.embed_out) > 0


def test_non_residual_dynamics_is_not_the_identity(key):
    """Zero-init applies only to the residual parameterisation."""
    model = xwm.families.jepa.action_world_model(
        key=key, action_dim=2,
        encoder=xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC),
        dynamics_kwargs={**PRED, "residual": False},
    )
    z = model.encode(jr.normal(key, (3, 16, 16)))
    assert not jnp.allclose(model.dynamics.eval_mode()(z, jnp.zeros((2,))), z, atol=1e-3)


def test_imagine_shapes_and_jit(key):
    model = _action_model(key)
    z0 = model.encode(jr.normal(key, (3, 16, 16)))
    actions = jr.normal(key, (6, 2))
    traj = eqx.filter_jit(model.imagine)(z0, actions)
    assert traj.shape == (6, *z0.shape)


def test_registry_builds_every_model():
    names = xwm.families.available()
    assert "jepa/image" in names and "jepa/video-lejepa" in names
    m = xwm.families.create(
        "jepa/image-lejepa", key=jr.PRNGKey(0), img_size=16, patch_size=4,
        n_targets=2, encoder_kwargs=ENC, predictor_kwargs=PRED,
    )
    assert isinstance(m, xwm.JEPA)
    with pytest.raises(KeyError, match="unknown model"):
        xwm.families.create("nope", key=jr.PRNGKey(0))


def test_both_model_families_share_the_inference_api(key):
    """A probe should not care which family trained the encoder.

    `JEPA` and `ActionWorldModel` must both expose `encode` -> (N, D) tokens and
    `embed` -> (D,) pooled, with matching shapes.
    """
    jepa = _image_model(key)
    action = _action_model(key)
    image = jr.normal(key, (3, 16, 16))
    for model in (jepa, action):
        tokens = model.encode(image)
        pooled = model.embed(image)
        assert tokens.ndim == 2 and tokens.shape[-1] == ENC["embed_dim"]
        assert pooled.shape == (ENC["embed_dim"],)
        assert jnp.allclose(pooled, jnp.mean(model.encoder.eval_mode()(image), axis=0), atol=1e-5)


def test_batched_apply_matches_a_single_call(key):
    """Chunking a dataset-wide map must not change the result."""
    f = jax.jit(lambda batch: jnp.sum(batch, axis=-1))
    x = jr.normal(key, (205, 8))
    assert jnp.allclose(xwm.core.batched_apply(f, x, batch_size=64), f(x), atol=1e-5)
    # A single chunk short-circuits, and must still be correct.
    assert jnp.allclose(xwm.core.batched_apply(f, x, batch_size=1000), f(x), atol=1e-5)


def test_batched_apply_handles_pytree_outputs(key):
    f = jax.jit(lambda b: {"sum": jnp.sum(b, -1), "first": b[:, 0]})
    x = jr.normal(key, (70, 4))
    out = xwm.core.batched_apply(f, x, batch_size=32)
    assert out["sum"].shape == (70,) and out["first"].shape == (70,)
    assert jnp.allclose(out["first"], x[:, 0])


def test_batched_apply_validates():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        xwm.core.batched_apply(lambda b: b, jnp.zeros((4, 2)), batch_size=0)
    with pytest.raises(ValueError, match="nothing to apply over"):
        xwm.core.batched_apply(lambda b: b, {})


def test_batched_apply_keeps_encoder_output_identical(key):
    """The real use: embedding a dataset in chunks must equal one big call."""
    encoder = xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=8, **ENC).eval_mode()
    images = jr.normal(key, (37, 3, 16, 16))
    whole = jax.vmap(encoder)(images)
    chunked = xwm.core.batched_apply(jax.jit(jax.vmap(encoder)), images, batch_size=8)
    assert chunked.shape == whole.shape
    assert jnp.allclose(chunked, whole, atol=1e-5)


# -- the planning contract (xwm.core.types.Plannable) -----------------------
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda k: _action_model(k), id="jepa/action"),
        pytest.param(
            lambda k: xwm.families.tdmpc2.tdmpc2(
                action_dim=2, observation="state", state_dim=6, latent_dim=32, hidden_dim=32, key=k
            ),
            id="tdmpc2",
        ),
        pytest.param(
            lambda k: xwm.families.muzero.muzero(
                n_actions=5, observation="state", state_dim=6, latent_dim=32, hidden_dim=32, key=k
            ),
            id="muzero",
        ),
    ],
)
def test_every_family_satisfies_the_plannable_protocol(build, key):
    """The benchmark layer talks to models through this and nothing else."""
    model = build(key)
    assert isinstance(model, xwm.core.Plannable)
    assert model.history == 1


def test_markov_initial_state_encodes_the_newest_frame(key):
    """With ``history == 1`` the latent state is just the last observation's encoding."""
    model = _action_model(key)
    frames = jr.normal(key, (1, 3, 16, 16))
    z = model.initial_state(frames)
    assert jnp.allclose(z, model.encode(frames[-1]))
    assert jnp.allclose(model.readout(z), z)  # identity readout
    assert jnp.allclose(model.goal_embedding(frames[-1]), z)


def test_initial_state_ignores_actions_when_markov(key):
    model = _action_model(key)
    frames = jr.normal(key, (1, 3, 16, 16))
    with_actions = model.initial_state(frames, jr.normal(jr.PRNGKey(9), (0, 2)))
    assert jnp.allclose(with_actions, model.initial_state(frames))


def test_plannable_state_feeds_a_planner_unchanged(key):
    """``initial_state`` -> ``dynamics_fn`` -> ``goal_cost(readout=)`` must compose."""
    model = _action_model(key).eval_mode()
    frames = jr.normal(key, (1, 3, 16, 16))
    z0 = model.initial_state(frames)
    cost = xwm.planning.goal_cost(model.goal_embedding(frames[-1]), readout=model.readout)
    planner = xwm.planning.CEM(3, 2, n_samples=32, n_elites=8, n_iters=2)
    plan = planner.plan(key, model.dynamics_fn(), z0, cost)
    assert plan.actions.shape == (3, 2)
    assert bool(jnp.isfinite(plan.cost))


def test_a_model_without_an_encoder_says_so(key):
    class Headless(xwm.core.WorldModel):
        def loss(self, batch, *, key, target=None):
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="has no encode"):
        Headless().initial_state(jnp.zeros((1, 3)))
