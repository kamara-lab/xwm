"""DINO-WM: a world model on frozen DINOv2 features.

Not a new architecture -- :class:`xwm.families.jepa.ARWorldModel` with a
pretrained encoder that stays frozen and a predictor over patch tokens rather
than a pooled vector. It is its own family because what it claims is its own:
that a general visual representation, trained on no robot data at all, is a good
enough state space to plan in.

>>> model = xwm.families.dinowm.dinowm(action_dim=7, img_size=224)   # doctest: +SKIP
"""

from .model import dinowm

__all__ = ["dinowm"]
