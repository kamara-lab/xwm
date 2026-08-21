"""Standard ViT and predictor widths, shared across families."""

from __future__ import annotations

#: Standard ViT widths.
VIT_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"embed_dim": 192, "depth": 12, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "huge": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
}

#: Matching predictor widths. Predictors are deliberately narrow -- see
#: :class:`JEPAPredictor`.
PREDICTOR_PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"pred_dim": 96, "depth": 4, "num_heads": 3},
    "small": {"pred_dim": 192, "depth": 6, "num_heads": 6},
    "base": {"pred_dim": 384, "depth": 6, "num_heads": 12},
    "large": {"pred_dim": 384, "depth": 12, "num_heads": 12},
    "huge": {"pred_dim": 384, "depth": 12, "num_heads": 12},
}


def preset(name: str, kind: str = "encoder") -> dict[str, int]:
    """Look up a size preset by name (``"tiny"`` ... ``"huge"``)."""
    table = VIT_PRESETS if kind == "encoder" else PREDICTOR_PRESETS
    if name not in table:
        raise KeyError(f"unknown {kind} preset {name!r}; choose from {sorted(table)}")
    return dict(table[name])
