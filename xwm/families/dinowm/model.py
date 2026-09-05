"""DINO-WM: dynamics on frozen DINOv2 patch features.

The other two windowed models in `xwm` learn their encoder. This one does not
learn anything about *seeing*: DINOv2 is frozen, and the only trained part is a
causal transformer that predicts what its patch grid will look like next.

Two claims come with that, and both are checkable here rather than taken on
faith. The first is that a general visual representation transfers to control
without task data -- the encoder never sees the robot. The second is that the
representation must be *spatial*: pooling DINOv2's patches into one vector
throws away where things are, which is most of what a manipulation task is
about. So :class:`~xwm.families.jepa.ARWorldModel` is built here with
``tokens="patch"``, and the action is concatenated to every token rather than
modulating the frame as a whole.

The price is tokens. At 224 pixels a frame is 256 of them, and a three-frame
window is 768 -- against three for the pooled models. That is the trade DINO-WM
makes and the reason LeWorldModel plans faster.

There is deliberately **no proprioception input**, which the paper does use on
some benchmarks. Adding it as a conditioning signal would break planning: the
planner proposes actions, not future joint angles, so a proprioceptive input has
to be *predicted* as part of the latent rather than supplied. That is a real
extension, not a flag, and it is not implemented here.

References
----------
Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features enable
Zero-shot Planning*, ICML 2025. arXiv:2411.04983.
"""

from __future__ import annotations

import jax.random as jr

from ...core.random import resolve_key
from ...core.types import PRNGKey
from ...dynamics.autoregressive import ARConditioning
from ...encoders.pretrained import PATCH_SIZE, dinov2
from ...encoders.vision import VisionEncoder
from ..jepa.autoregressive import ARWorldModel, _build

__all__ = ["dinowm"]


def dinowm(
    *,
    key: PRNGKey | None = None,
    action_dim: int,
    history: int = 3,
    encoder: VisionEncoder | None = None,
    size: str = "small",
    img_size: int | tuple[int, int] = 224,
    registers: bool = False,
    freeze_encoder: bool = True,
    action_scale: float = 1.0,
    conditioning: ARConditioning = "concat",
    dynamics_kwargs: dict | None = None,
    **model_kwargs,
) -> ARWorldModel:
    """Assemble DINO-WM: a frozen DINOv2 encoder and a causal patch-level predictor.

    Args:
        action_dim: width of one action.
        history: frames the predictor conditions on. Three in the paper.
        encoder: an encoder to build on, instead of downloading DINOv2. Any
            :class:`~xwm.encoders.VisionEncoder` works, which is how a
            JEPA-pretrained `xwm` encoder is substituted -- and how the tests
            avoid the network.
        size: which DINOv2 -- ``"small"``, ``"base"``, ``"large"``, ``"giant"``.
        img_size: must be a multiple of 14, DINOv2's patch size.
        registers: load the register-token variant of the weights.
        freeze_encoder: keep the pretrained features fixed. On by default, and
            the reason there is no collapse term: an encoder that cannot move
            cannot degrade its own targets.

    Needs ``pip install xwm[pretrained]`` unless ``encoder`` is supplied.

    Example:
        >>> model = xwm.families.dinowm.dinowm(action_dim=7, img_size=224)   # doctest: +SKIP
    """
    k_enc, k_rest = jr.split(resolve_key(key), 2)
    if encoder is None:
        height = img_size if isinstance(img_size, int) else img_size[0]
        width = img_size if isinstance(img_size, int) else img_size[1]
        if height % PATCH_SIZE or width % PATCH_SIZE:
            raise ValueError(
                f"DINO-WM's encoder uses {PATCH_SIZE}-pixel patches, so img_size "
                f"{(height, width)} must be a multiple of {PATCH_SIZE}. Set the task's "
                f"image_size to {PATCH_SIZE * max(1, round(height / PATCH_SIZE))}."
            )
        encoder = dinov2(size, img_size=img_size, registers=registers)
        del k_enc

    model_kwargs.setdefault("collapse", "none")
    model_kwargs.setdefault("detach_target", True)
    return _build(
        key=k_rest,
        action_dim=action_dim,
        history=history,
        encoder=encoder,
        size="base",
        img_size=img_size,
        patch_size=PATCH_SIZE,
        in_channels=3,
        tokens="patch",
        latent_dim=None,
        action_scale=action_scale,
        conditioning=conditioning,
        encoder_kwargs=None,
        dynamics_kwargs={"depth": 6, "num_heads": 16, **(dynamics_kwargs or {})},
        action_decoder_kwargs=None,
        with_action_decoder=False,
        freeze_encoder=freeze_encoder,
        **model_kwargs,
    )
