"""Why doesn't a JEPA collapse? Compare the answers head to head.

Predicting a representation from a representation admits a trivial solution:
emit the same vector for every input, and the prediction loss goes to zero
without the model having learned anything. Every JEPA needs a countermeasure,
and this script runs four of them on identical data and architecture:

* ``ema``    -- I-JEPA / V-JEPA: targets from a slowly-moving teacher.
* ``sigreg`` -- LeJEPA: one encoder, no stop-gradient, and a distributional
                penalty that forbids the constant solution outright.
* ``vicreg`` -- variance and covariance penalties on the embeddings.
* ``none``   -- the control. Watch it collapse.

The prediction loss is *not* the metric to compare. A collapsed model has the
best loss of all. Read the diagnostics instead:

* ``feature_std``  per-dimension spread across inputs; -> 0 means collapsed.
* ``mean_cosine``  average similarity between inputs; -> 1 means collapsed.
* ``rankme``       effective rank of the embedding spectrum; higher is better.

Note that the linear probe is *not* a collapse detector: it standardises
features first, so it happily amplifies a nearly-dead signal back to full
scale. That is exactly why the diagnostics above exist.

What to expect: the control has the *best* prediction loss and a completely
dead representation. On a dataset this small the EMA teacher collapses too --
its mechanism depends on data diversity and a long schedule, which is the
fragility LeJEPA sets out to remove. SIGReg and VICReg then fail in different
directions: SIGReg drives `mean_cosine` toward 0, while VICReg's covariance
term maximises `rankme` yet leaves samples pointing broadly the same way.

Artifacts: a bar-chart panel per diagnostic, overlaid embedding spectra,
training curves, and the comparison table as Markdown/LaTeX/CSV.

Run: python examples/03_collapse_strategies.py
"""

import jax
import jax.random as jr
from _common import report, setting, setup

import xwm

IMG_SIZE = setting("IMG_SIZE", 32)
PATCH_SIZE = setting("PATCH_SIZE", 4)
STEPS = setting("STEPS", 400)
BATCH = setting("BATCH", 32)
ENC = {
    "depth": setting("ENC_DEPTH", 4),
    "embed_dim": setting("ENC_DIM", 128),
    "num_heads": setting("ENC_HEADS", 4),
}
PRED = {"depth": 3, "pred_dim": 64, "num_heads": 4}

CONFIGS = [
    ("ema (I-JEPA)", {"collapse": "ema"}),
    ("sigreg w=5 (LeJEPA)", {"collapse": "sigreg", "reg_weight": 5.0}),
    ("sigreg w=20 (LeJEPA)", {"collapse": "sigreg", "reg_weight": 20.0}),
    ("vicreg w=1", {"collapse": "vicreg", "reg_weight": 1.0}),
    ("none (control)", {"collapse": "none"}),
]


def run(key, data, evaluation, *, collapse, reg_weight=1.0):
    base = xwm.families.jepa.ijepa(
        key=key, img_size=IMG_SIZE, patch_size=PATCH_SIZE, n_targets=4,
        encoder_kwargs=ENC, predictor_kwargs=PRED,
    )
    model = xwm.JEPA(
        base.encoder, base.predictor, base.mask_sampler, input_key="image",
        collapse=collapse, loss_kind="smooth_l1", reg_weight=reg_weight, n_proj=256,
    )
    trainer = xwm.training.Trainer(
        model,
        xwm.training.adamw(xwm.training.cosine_warmup(1e-3, STEPS, warmup_steps=STEPS // 10)),
        ema_momentum=xwm.training.ema_momentum(STEPS, base=0.996),
    )
    batches = xwm.data.iter_batches({"image": data["image"]}, BATCH, key=key, epochs=None)
    state, history = trainer.fit(batches, steps=STEPS, key=key, log_every=STEPS)

    embed = jax.jit(jax.vmap(state.model.embed))
    z = embed(evaluation["image"])
    diagnostics = xwm.metrics.collapse_report(z)
    probe = xwm.metrics.ridge_probe(
        embed(data["image"]), data["position"], z, evaluation["position"]
    )
    return {
        "pred_loss": history[-1]["loss_pred"],
        "reg": history[-1]["loss_reg"],
        **{k: float(v) for k, v in diagnostics.items()},
        "probe_r2": float(probe["r2"]),
        "history": history,
        "embeddings": z,
    }


def main():
    out = setup("03_collapse_strategies", n_series=len(CONFIGS))
    key = jr.PRNGKey(0)
    k_data, k_run = jr.split(key)
    world = xwm.data.SpriteWorld(IMG_SIZE, n_distractors=2)
    data = xwm.data.sprite_images(jr.fold_in(k_data, 0), 2048, world=world)
    evaluation = xwm.data.sprite_images(jr.fold_in(k_data, 1), 512, world=world)
    dim = ENC["embed_dim"]

    header = (
        f"{'strategy':<22}{'pred_loss':>10}{'reg':>9}"
        f"{'feat_std':>10}{'mean_cos':>10}{'rankme':>9}{'probe_r2':>10}"
    )
    print(f"{STEPS} steps, {dim}-dim embeddings, 2048 sprite images\n")
    print(header)
    print("-" * len(header))
    results = {}
    for name, cfg in CONFIGS:
        r = run(k_run, data, evaluation, **cfg)
        results[name] = r
        print(
            f"{name:<22}{r['pred_loss']:>10.4f}{r['reg']:>9.4f}"
            f"{r['feature_std']:>10.4f}{r['mean_cosine']:>+10.3f}"
            f"{r['rankme']:>7.1f}/{dim}{r['probe_r2']:>10.3f}",
            flush=True,
        )
    print(
        "\nfeat_std -> 0 and mean_cos -> 1 both indicate collapse; "
        "rankme is the effective\nrank out of "
        f"{dim}. Note that pred_loss is *lowest* for the most collapsed runs."
    )

    # -- artifacts ----------------------------------------------------------
    print("\nartifacts:")
    import matplotlib.pyplot as plt

    names = list(results)
    # One panel per diagnostic: they have different units and different ideal
    # directions, so a single shared axis would be meaningless.
    panels = [
        ("mean_cosine", "mean cosine (0 is good, 1 is collapsed)"),
        ("feature_std", "feature std (higher is better)"),
        ("rankme", f"RankMe out of {dim} (higher is better)"),
        ("pred_loss", "prediction loss (NOT a quality measure)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for cell, (metric, label) in zip(axes.ravel(), panels, strict=True):
        xwm.plots.plot_bars(
            names, [results[n][metric] for n in names], ax=cell, ylabel=label, title=metric
        )
    fig.suptitle(
        "Anti-collapse strategies: the best prediction loss belongs to the worst representation",
        fontsize=11,
    )
    fig.tight_layout()
    report(xwm.plots.save_figure(fig, out / "diagnostics.png"))

    ax = xwm.plots.plot_spectra(
        {n: results[n]["embeddings"] for n in names},
        title="Embedding spectra by anti-collapse strategy",
    )
    report(xwm.plots.save_figure(ax.figure, out / "spectra.png"))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, name in enumerate(names):
        history = results[name]["history"]
        ax.plot(
            [r["step"] for r in history],
            [r["loss_pred"] for r in history],
            label=name,
            markevery=max(1, len(history) // 10),
            # Five strategies exceeds what blue-orange can separate, so this
            # many-way comparison stays on viridis and keeps the same colour per
            # strategy as the diagnostic panels above.
            **xwm.plots.series_style(i, len(names), xwm.plots.MAGNITUDE_PALETTE),
        )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("prediction loss")
    ax.set_title("Prediction loss alone cannot rank these")
    ax.legend(loc="lower left")
    report(xwm.plots.save_figure(fig, out / "training_curves.png"))

    rows = [
        [
            name,
            results[name]["pred_loss"],
            results[name]["reg"],
            results[name]["feature_std"],
            results[name]["mean_cosine"],
            results[name]["rankme"],
            results[name]["probe_r2"],
        ]
        for name in names
    ]
    report(
        xwm.plots.save_table(
            out / "results",
            ["strategy", "pred_loss", "reg", "feature_std", "mean_cosine",
             f"rankme (of {dim})", "probe_r2"],
            rows,
            caption=(
                "Anti-collapse strategies on 2048 sprite images, "
                f"{STEPS} steps, {dim}-dim embeddings. Lower prediction loss "
                "does not mean a better representation."
            ),
            label="tab:collapse",
        )
    )


if __name__ == "__main__":
    main()
