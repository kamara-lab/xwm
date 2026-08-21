"""The training loop: does it decrease the loss, respect freezing, and resume?"""

import jax.numpy as jnp
import jax.random as jr
import pytest

import xwm

ENC = {"depth": 2, "embed_dim": 64, "num_heads": 4}
PRED = {"depth": 2, "pred_dim": 32, "num_heads": 4}


def _model(key, *, encoder_kwargs=None, **kw):
    return xwm.families.jepa.ijepa(
        key=key, img_size=16, patch_size=4, n_targets=2,
        encoder_kwargs={**ENC, **(encoder_kwargs or {})}, predictor_kwargs=PRED, **kw
    )


def _data(key, n=64):
    return {"image": jr.normal(key, (n, 3, 16, 16))}


def _batches(key, n=64, batch=8):
    return xwm.data.iter_batches(_data(key, n), batch, key=key, epochs=None)


def test_fit_decreases_the_loss(key):
    model = _model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3), ema_momentum=0.99)
    state, history = trainer.fit(_batches(key), steps=40, key=key, log_every=10)
    assert len(history) >= 4
    assert history[-1]["loss"] < history[0]["loss"]
    assert int(state.step) == 40


def test_history_rows_carry_step_and_metrics(key):
    trainer = xwm.training.Trainer(_model(key), xwm.training.adamw(1e-3))
    _, history = trainer.fit(_batches(key), steps=5, key=key, log_every=1)
    assert [row["step"] for row in history] == [1, 2, 3, 4, 5]
    assert {"loss", "loss_pred", "grad_norm"} <= set(history[0])
    assert isinstance(history[0]["step"], int)
    assert all(isinstance(v, float) for k, v in history[0].items() if k != "step")


def test_ema_target_lags_the_student(key):
    model = _model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-2), ema_momentum=0.9)
    state, _ = trainer.fit(_batches(key), steps=10, key=key, log_every=0)

    def w(m):
        return m.encoder.blocks.blocks[0].attn.qkv.weight

    start = w(model)
    assert float(jnp.linalg.norm(w(state.model) - start)) > 0  # student moved
    assert float(jnp.linalg.norm(w(state.target) - start)) > 0  # teacher followed
    # ...but the teacher moved strictly less far than the student.
    assert float(jnp.linalg.norm(w(state.target) - start)) < float(
        jnp.linalg.norm(w(state.model) - start)
    )


def test_no_target_allocated_without_ema(key):
    model = xwm.families.jepa.lejepa(
        key=key, img_size=16, patch_size=4, n_targets=2,
        encoder_kwargs=ENC, predictor_kwargs=PRED, reg_weight=1.0,
    )
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    state = trainer.init()
    assert state.target is None
    state, history = trainer.fit(_batches(key), steps=20, key=key, log_every=10)
    assert state.target is None
    assert bool(jnp.isfinite(jnp.asarray(history[-1]["loss"])))


def test_momentum_schedule_is_followed(key):
    schedule = xwm.training.ema_momentum(100, base=0.9, final=1.0)
    trainer = xwm.training.Trainer(_model(key), xwm.training.adamw(1e-3), ema_momentum=schedule)
    assert trainer.momentum_at(0) == pytest.approx(0.9)
    assert trainer.momentum_at(100) == pytest.approx(1.0)
    assert trainer.momentum_at(50) == pytest.approx(0.95)
    flat = xwm.training.Trainer(_model(key), xwm.training.adamw(1e-3), ema_momentum=0.996)
    assert flat.momentum_at(0) == flat.momentum_at(999) == pytest.approx(0.996)


def test_frozen_parameters_never_change(key):
    model = xwm.families.jepa.action_world_model(
        key=key, action_dim=2,
        encoder=xwm.encoders.ImageEncoder(key=key, img_size=16, patch_size=4, **ENC),
        dynamics_kwargs=PRED, freeze_encoder=True,
    )
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-2))
    data = {
        "video": jr.normal(key, (16, 4, 3, 16, 16)),
        "action": jr.normal(key, (16, 3, 2)),
    }
    state, history = trainer.fit(
        xwm.data.iter_batches(data, 4, key=key, epochs=None), steps=20, key=key, log_every=10
    )

    def enc_w(m):
        return m.encoder.blocks.blocks[0].attn.qkv.weight

    assert jnp.array_equal(enc_w(state.model), enc_w(model)), "frozen encoder changed"
    dyn = state.model.dynamics.blocks.blocks[0].attn.qkv.weight
    assert not jnp.array_equal(dyn, model.dynamics.blocks.blocks[0].attn.qkv.weight)
    assert history[-1]["loss"] < history[0]["loss"]


def test_resume_from_state(key):
    model = _model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    state, _ = trainer.fit(_batches(key), steps=10, key=key, log_every=0)
    resumed, history = trainer.fit(
        _batches(key), steps=5, key=jr.PRNGKey(1), state=state, log_every=1
    )
    assert int(resumed.step) == 15
    assert history[0]["step"] == 11


def test_evaluate_runs_in_eval_mode(key):
    model = _model(key, encoder_kwargs={"dropout": 0.5})
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    state = trainer.init()
    batches = [{"image": jr.normal(key, (8, 3, 16, 16))}]
    a = trainer.evaluate(state, batches, key=jr.PRNGKey(1))
    b = trainer.evaluate(state, batches, key=jr.PRNGKey(1))
    assert a["loss"] == pytest.approx(b["loss"])  # deterministic given the key
    assert set(a) >= {"loss", "loss_pred"}


def test_grad_clipping_is_applied(key):
    """A tiny clip norm must bound the update size."""
    model = _model(key)
    loose = xwm.training.Trainer(model, xwm.training.adamw(1e-2, grad_clip=None))
    tight = xwm.training.Trainer(model, xwm.training.adamw(1e-2, grad_clip=1e-6))
    batch = {"image": jr.normal(key, (8, 3, 16, 16))}

    def moved(trainer):
        state, _ = trainer.step(trainer.init(), batch, key)
        return float(
            jnp.linalg.norm(
                state.model.encoder.blocks.blocks[0].attn.qkv.weight
                - model.encoder.blocks.blocks[0].attn.qkv.weight
            )
        )

    assert moved(tight) < moved(loose)


def test_global_norm():
    assert float(xwm.training.global_norm({"a": jnp.array([3.0, 4.0])})) == pytest.approx(5.0)
    assert float(xwm.training.global_norm({})) == 0.0


def test_print_metrics_callback(key, capsys):
    trainer = xwm.training.Trainer(_model(key), xwm.training.adamw(1e-3))
    trainer.fit(
        _batches(key), steps=3, key=key, log_every=1,
        callbacks=[xwm.training.print_metrics(keys=["loss"])],
    )
    out = capsys.readouterr().out
    assert out.count("loss=") == 3


def test_checkpoint_roundtrip(key, tmp_path):
    model = _model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    state, _ = trainer.fit(_batches(key), steps=5, key=key, log_every=0)

    path = tmp_path / "model.eqx"
    xwm.tools.save(path, state.model, config={"img_size": 16})
    fresh = _model(jr.PRNGKey(999))
    assert not jnp.array_equal(
        fresh.encoder.blocks.blocks[0].attn.qkv.weight,
        state.model.encoder.blocks.blocks[0].attn.qkv.weight,
    )
    loaded = xwm.tools.load(path, fresh)
    assert jnp.array_equal(
        loaded.encoder.blocks.blocks[0].attn.qkv.weight,
        state.model.encoder.blocks.blocks[0].attn.qkv.weight,
    )
    assert xwm.tools.load_config(path) == {"img_size": 16}


def test_state_checkpoint_roundtrip(key, tmp_path):
    model = _model(key)
    trainer = xwm.training.Trainer(model, xwm.training.adamw(1e-3))
    state, _ = trainer.fit(_batches(key), steps=5, key=key, log_every=0)
    path = tmp_path / "state.eqx"
    xwm.tools.save_state(path, state)
    restored = xwm.tools.load_state(path, trainer.init())
    assert int(restored.step) == 5
    assert jnp.array_equal(
        restored.target.encoder.blocks.blocks[0].attn.qkv.weight,
        state.target.encoder.blocks.blocks[0].attn.qkv.weight,
    )


def test_load_config_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        xwm.tools.load_config(tmp_path / "nothing.eqx")


def test_summary_and_counts(key):
    model = _model(key)
    text = xwm.tools.summary(model, max_depth=2)
    assert "JEPA" in text and "encoder" in text and "predictor" in text
    assert xwm.tools.count_params(model) == model.n_params
    assert xwm.tools.param_bytes(model) == model.n_params * 4  # float32
