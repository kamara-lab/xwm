"""The JEPA world model: predict masked latents, and don't collapse.

Every model in :mod:`xwm.image` and :mod:`xwm.video` is this one class with a
different tokeniser and mask sampler. It has exactly two moving parts:

**What to predict.** A mask sampler splits the token grid into a visible
context and one or more target blocks. The context encoder sees only visible
tokens; the predictor is asked for the targets' *embeddings*, given nothing
about their content but their position.

**Why it doesn't collapse.** Predicting a representation from a representation
has a trivial solution -- emit a constant. The ``collapse`` argument selects the
countermeasure, and this is the axis the literature actually disagrees on:

``"ema"``
    Targets come from a slowly-moving copy of the encoder, with gradients cut.
    The asymmetry between a fast student and a slow teacher is what keeps the
    constant solution from being reachable. I-JEPA, V-JEPA.
``"sigreg"``
    Targets come from the *same* encoder, gradients flow through both branches,
    and a distributional penalty forbids the constant solution outright.
    No teacher, no stop-gradient, no schedule -- one coefficient. LeJEPA.
``"vicreg"``
    Variance and covariance penalties on the embeddings; constrains the first
    two moments where SIGReg constrains the whole distribution.
``"none"``
    No countermeasure. Useful only to watch collapse happen -- which it will.

References
----------
Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
Predictive Architecture* (I-JEPA), CVPR 2023. arXiv:2301.08243

Bardes et al., *Revisiting Feature Prediction for Learning Visual
Representations from Video* (V-JEPA), 2024. arXiv:2404.08471

Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning
Without the Heuristics*, 2025.

Grill et al., *Bootstrap Your Own Latent* (BYOL) -- the EMA-teacher mechanism
``collapse="ema"`` implements, NeurIPS 2020. arXiv:2006.07733

Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization
for Self-Supervised Learning*, ICLR 2022. arXiv:2105.04906
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ...core.ema import stop_gradient
from ...core.module import WorldModel, vmap_apply
from ...core.types import Array, Batch, Metrics, PRNGKey
from ...encoders.vision import VisionEncoder
from ...masking.base import MaskBatch
from ...objectives.prediction import LossKind, prediction_loss
from ...objectives.regularizers import covariance_loss, variance_loss
from ...objectives.sigreg import Statistic, sigreg
from .predictor import JEPAPredictor

Collapse = Literal["ema", "sigreg", "vicreg", "none"]

__all__ = ["JEPA", "Collapse"]


class JEPA(WorldModel):
    """Joint-embedding predictive architecture over a token grid.

    Args:
        encoder: context encoder; also the teacher unless ``collapse="ema"``.
        predictor: maps context tokens to target-position embeddings.
        mask_sampler: callable ``key -> MaskBatch`` (see :mod:`xwm.masking`).
        input_key: which field of the batch holds the input tensor.
        collapse: anti-collapse strategy; see the module docstring.
        loss_kind: how predicted and target embeddings are compared.
        normalize_target: LayerNorm targets before comparing (V-JEPA does).
        reg_weight: coefficient on the ``sigreg`` / ``vicreg`` penalty. This is
            LeJEPA's single hyperparameter.
        n_proj, statistic, n_nodes, sigma: SIGReg settings.
    """

    encoder: VisionEncoder
    predictor: JEPAPredictor
    mask_sampler: eqx.Module
    input_key: str = eqx.field(static=True)
    collapse: Collapse = eqx.field(static=True)
    loss_kind: LossKind = eqx.field(static=True)
    normalize_target: bool = eqx.field(static=True)
    reg_weight: float = eqx.field(static=True)
    n_proj: int = eqx.field(static=True)
    statistic: Statistic = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    sigma: float = eqx.field(static=True)

    def __init__(
        self,
        encoder: VisionEncoder,
        predictor: JEPAPredictor,
        mask_sampler: eqx.Module,
        *,
        input_key: str = "image",
        collapse: Collapse = "ema",
        loss_kind: LossKind = "smooth_l1",
        normalize_target: bool = False,
        reg_weight: float = 1.0,
        n_proj: int = 256,
        statistic: Statistic = "epps_pulley",
        n_nodes: int = 32,
        sigma: float = 1.0,
    ):
        if encoder.embed_dim != predictor.embed_dim:
            raise ValueError(
                f"encoder width {encoder.embed_dim} != predictor width {predictor.embed_dim}"
            )
        if tuple(encoder.grid) != tuple(predictor.grid):
            raise ValueError(f"grid mismatch: {encoder.grid} vs {predictor.grid}")
        self.encoder = encoder
        self.predictor = predictor
        self.mask_sampler = mask_sampler
        self.input_key = input_key
        self.collapse = collapse
        self.loss_kind = loss_kind
        self.normalize_target = normalize_target
        self.reg_weight = reg_weight
        self.n_proj = n_proj
        self.statistic = statistic
        self.n_nodes = n_nodes
        self.sigma = sigma
        # Only the EMA strategy needs the trainer to maintain a teacher.
        self.uses_target = collapse == "ema"

    # -- inference -----------------------------------------------------------
    def encode(self, x: Array, *, keep: Array | None = None, key: PRNGKey | None = None) -> Array:
        """Encode one sample to ``(N, D)`` tokens. This is the transferable part."""
        return self.encoder(x, keep=keep, key=key)

    def embed(self, x: Array) -> Array:
        """Mean-pooled ``(D,)`` representation of one sample, in eval mode."""
        return jnp.mean(self.encoder.eval_mode()(x), axis=0)

    # -- training ------------------------------------------------------------
    def prepare_batch(self, batch: Batch, key: PRNGKey) -> Batch:
        """Attach a freshly sampled mask. Runs on the host, outside ``jit``."""
        if "masks" in batch:
            return batch
        return {**batch, "masks": self.mask_sampler(key)}

    def predict(
        self,
        context: Array,
        masks: MaskBatch,
        *,
        key: PRNGKey | None = None,
    ) -> Array:
        """Predict every target block from one sample's context tokens.

        Returns ``(M, K_tgt, D)``. Each block gets its own mask token (cycling
        if there are more blocks than tokens), so the predictor can distinguish
        the concurrent prediction problems it is being asked to solve.
        """
        n_mask_tokens = self.predictor.mask_tokens.shape[0]
        ids = jnp.arange(masks.n_targets) % n_mask_tokens
        return jax.vmap(
            lambda tgt, i: self.predictor(context, masks.context, tgt, mask_index=i, key=key)
        )(masks.targets, ids)

    def loss(
        self,
        batch: Batch,
        *,
        key: PRNGKey,
        target: JEPA | None = None,
    ) -> tuple[Array, Metrics]:
        if "masks" not in batch:
            raise KeyError(
                "batch has no 'masks'; call model.prepare_batch(batch, key) first "
                "(xwm.training.Trainer does this for you)"
            )
        x, masks = batch[self.input_key], batch["masks"]
        # The teacher branch needs no key: it runs in eval mode by construction.
        k_ctx, k_pred, k_reg = jr.split(key, 3)

        # Context branch: only visible tokens are computed at all.
        context = vmap_apply(
            lambda xi, key: self.encoder(xi, keep=masks.context, key=key), x, key=k_ctx
        )  # (B, K_ctx, D)

        # Target branch: the whole grid, from the teacher. Run in eval mode so
        # dropout noise is never mistaken for something to predict.
        teacher = (target.encoder if self.uses_target else self.encoder).eval_mode()
        full = jax.vmap(teacher)(x)  # (B, N, D)
        if self.uses_target:
            full = stop_gradient(full)
        targets = full[:, masks.targets]  # (B, M, K_tgt, D)

        pred = vmap_apply(
            lambda c, key: self.predict(c, masks, key=key), context, key=k_pred
        )  # (B, M, K_tgt, D)

        loss_pred = prediction_loss(
            pred,
            targets,
            kind=self.loss_kind,
            normalize_target=self.normalize_target,
        )
        metrics: Metrics = {
            "loss_pred": loss_pred,
            "embed_std": jnp.mean(jnp.std(full.reshape(-1, full.shape[-1]), axis=0)),
        }

        reg = jnp.zeros(())
        if self.collapse == "sigreg":
            # Applied to both branches: the full-grid embeddings and the
            # context encoding are different views of the same encoder, and
            # constraining only one leaves the other free to degenerate.
            def penalty(z, k):
                return sigreg(
                    z,
                    k,
                    n_proj=self.n_proj,
                    statistic=self.statistic,
                    n_nodes=self.n_nodes,
                    sigma=self.sigma,
                )

            k_full, k_context = jr.split(k_reg)
            reg = 0.5 * (penalty(full, k_full) + penalty(context, k_context))
        elif self.collapse == "vicreg":
            # The prediction loss already supplies VICReg's invariance term, so
            # only the variance and covariance terms are added here.
            flat = full.reshape(-1, full.shape[-1])
            reg = variance_loss(flat) + covariance_loss(flat)

        metrics["loss_reg"] = reg
        total = loss_pred + self.reg_weight * reg
        metrics["loss"] = total
        return total, metrics
