"""Visualising frames, masks and latents."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..core.types import Array
from ._backend import pyplot
from .style import SEQUENTIAL_CMAP, plot_style


def _to_hwc(frame: Array) -> np.ndarray:
    """``(C, H, W)`` -> ``(H, W, C)`` or ``(H, W)``, clipped for display."""
    arr = np.asarray(jnp.clip(frame, 0.0, 1.0))
    if arr.ndim == 2:
        return arr
    return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)


def plot_frames(
    frames: Array,
    *,
    titles=None,
    max_frames: int = 12,
    scale: float = 1.3,
    suptitle: str | None = None,
):
    """Draw a strip of ``(T, C, H, W)`` frames.

    ``scale`` is inches per panel. Text scales with it: a strip sized to show a
    768 px render natively is ~30 inches wide, where 8 pt titles are unreadable
    and a fixed-height suptitle lands on top of them.
    """
    plt = pyplot()
    n = min(frames.shape[0], max_frames)
    title_size = max(8.0, 6.0 * scale)
    suptitle_size = max(10.0, 7.0 * scale)
    height = scale + 0.4
    with plot_style(1):
        fig, axes = plt.subplots(1, n, figsize=(scale * n, height))
        axes = np.atleast_1d(axes)
        for i in range(n):
            axes[i].imshow(_to_hwc(frames[i]), cmap=SEQUENTIAL_CMAP)
            axes[i].axis("off")
            axes[i].set_title(
                str(titles[i]) if titles is not None else f"t={i}", fontsize=title_size
            )
        if suptitle:
            fig.suptitle(suptitle, fontsize=suptitle_size)
            # Reserve the suptitle's own height, in figure fractions, so it
            # cannot overlap the panel titles at any aspect ratio.
            reserved = min(0.4, (suptitle_size * 2.0 / 72.0) / height)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - reserved))
        else:
            fig.tight_layout()
    return fig


def plot_rollout(true_frames: Array, imagined: Array | None = None, *, max_frames: int = 10):
    """Compare a real trajectory against an imagined one, frame by frame.

    ``imagined`` is optional because a latent world model has nothing to render;
    pass decoded frames only if you have a decoder. Otherwise use this for the
    ground-truth strip and report latent distances numerically.
    """
    plt = pyplot()
    rows = 1 if imagined is None else 2
    n = min(true_frames.shape[0], max_frames)
    with plot_style(1):
        fig, axes = plt.subplots(rows, n, figsize=(1.3 * n, 1.5 * rows), squeeze=False)
        for i in range(n):
            axes[0][i].imshow(_to_hwc(true_frames[i]), cmap=SEQUENTIAL_CMAP)
            axes[0][i].set_title(f"t={i}", fontsize=8)
            if imagined is not None:
                axes[1][i].imshow(_to_hwc(imagined[i]), cmap=SEQUENTIAL_CMAP)
        for row, name in zip(axes, ("observed", "imagined"), strict=False):
            for j, cell in enumerate(row):
                cell.set_xticks([])
                cell.set_yticks([])
                for spine in cell.spines.values():
                    spine.set_visible(False)
                if j == 0:
                    cell.set_ylabel(name, fontsize=9)
        fig.tight_layout()
    return fig


def plot_mask(frame: Array, mask_idx: Array, grid: tuple[int, int], *, ax=None):
    """Overlay a token mask on an image, dimming the masked patches.

    Args:
        frame: ``(C, H, W)`` image.
        mask_idx: flat token indices considered *visible*.
        grid: ``(gh, gw)`` patch grid the indices refer to.
    """
    plt = pyplot()
    gh, gw = grid
    visible = np.zeros(gh * gw, dtype=bool)
    visible[np.asarray(mask_idx)] = True
    img = _to_hwc(frame)
    h, w = img.shape[:2]
    alpha = np.kron(visible.reshape(gh, gw), np.ones((h // gh, w // gw)))
    shaded = img * (0.25 + 0.75 * alpha[..., None] if img.ndim == 3 else 0.25 + 0.75 * alpha)
    with plot_style(1):
        ax = ax or plt.subplots(figsize=(3, 3))[1]
        ax.imshow(np.clip(shaded, 0, 1), cmap=SEQUENTIAL_CMAP)
        ax.set_title(f"{int(visible.sum())}/{gh * gw} tokens visible", fontsize=9)
        ax.axis("off")
    return ax


def plot_latent_pca(
    z: Array,
    *,
    labels: Array | None = None,
    ax=None,
    label_name: str = "value",
    title: str | None = None,
):
    """Scatter embeddings on their first two principal components.

    ``labels`` is a *continuous* quantity (an agent coordinate, a joint angle),
    so it is encoded with the viridis ramp and a colourbar -- magnitude, not
    identity. If the structure the encoder learned corresponds to that
    quantity, the scatter shows a smooth gradient rather than a blob.
    """
    plt = pyplot()
    flat = z.reshape(-1, z.shape[-1])
    if labels is not None:
        n_labels = np.asarray(labels).reshape(-1).shape[0]
        if n_labels != flat.shape[0]:
            raise ValueError(
                f"got {n_labels} labels for {flat.shape[0]} points (input shape "
                f"{tuple(z.shape)} flattens to {flat.shape[0]} rows). Pool token "
                "sequences to one vector per sample first, e.g. z.mean(axis=-2)."
            )
    centred = flat - jnp.mean(flat, axis=0, keepdims=True)
    _, _, vt = jnp.linalg.svd(centred, full_matrices=False)
    proj = np.asarray(centred @ vt[:2].T)
    with plot_style(1):
        ax = ax or plt.subplots(figsize=(5, 4.2))[1]
        if labels is None:
            from .style import ACCENT

            # No labels means no magnitude to encode, so this is one mark rather
            # than a ramp sampled at one point: the brand accent, not viridis.
            ax.scatter(proj[:, 0], proj[:, 1], s=14, color=ACCENT, alpha=0.75)
        else:
            scatter = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                c=np.asarray(labels).reshape(-1),
                s=14,
                cmap=SEQUENTIAL_CMAP,
                alpha=0.9,
            )
            ax.figure.colorbar(scatter, ax=ax, label=label_name)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(axis="both", visible=True)
        if title:
            ax.set_title(title)
    return ax
