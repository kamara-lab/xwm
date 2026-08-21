"""Writing figures and animated GIFs to disk.

Examples and papers need artifacts, not just numbers on a terminal. These
helpers keep that concern out of the example scripts: a rollout is a
``(T, C, H, W)`` array, and turning it into a legible GIF -- upscaled, tiled
side by side, labelled -- should not be twenty lines at every call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..core.types import Array


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "GIF export needs Pillow; install it with `pip install xwm[plots]`"
        ) from exc
    return Image, ImageDraw, ImageFont


def frames_to_uint8(frames: Array) -> np.ndarray:
    """Normalise a frame sequence to ``(T, H, W, 3)`` ``uint8``.

    Accepts channels-first ``(T, C, H, W)`` or channels-last ``(T, H, W, C)``,
    grayscale or RGB, float in ``[0, 1]`` or already ``uint8``.
    """
    arr = np.asarray(frames)
    if arr.ndim == 3:  # (T, H, W) grayscale
        arr = arr[:, None]
    if arr.ndim != 4:
        raise ValueError(f"expected a 3-D or 4-D frame sequence, got shape {arr.shape}")

    channel_counts = (1, 3, 4)
    first, last = arr.shape[1], arr.shape[-1]
    first_ok, last_ok = first in channel_counts, last in channel_counts
    if not (first_ok or last_ok):
        raise ValueError(
            f"cannot find a channel axis in shape {arr.shape}; expected 1, 3 or 4 "
            "channels at axis 1 (C, H, W) or axis 3 (H, W, C)"
        )
    # When both axes could be channels (e.g. a 3-pixel-wide image), prefer the
    # smaller one; xwm produces channels-first, so that is the right tiebreak.
    if first_ok and (not last_ok or first <= last):
        arr = arr.transpose(0, 2, 3, 1)

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr[..., :3]


def upscale(frames: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbour integer upscale, so pixels stay crisp rather than blurred."""
    if factor <= 1:
        return frames
    return np.repeat(np.repeat(frames, factor, axis=1), factor, axis=2)


def tile_frames(
    sequences: Sequence[Array],
    *,
    labels: Sequence[str] | None = None,
    scale: int = 4,
    pad: int = 2,
    label_height: int = 14,
) -> np.ndarray:
    """Lay several frame sequences side by side into one sequence.

    Sequences of differing length are truncated to the shortest, so a comparison
    strip never silently pairs frame 5 of one rollout with frame 9 of another.
    """
    Image, ImageDraw, ImageFont = _pillow()
    panels = [upscale(frames_to_uint8(s), scale) for s in sequences]
    n = min(p.shape[0] for p in panels)
    height = max(p.shape[1] for p in panels)
    panels = [p[:n] for p in panels]

    header = label_height if labels is not None else 0
    widths = [p.shape[2] for p in panels]
    total_width = sum(widths) + pad * (len(panels) - 1)
    out = np.full((n, height + header, total_width, 3), 255, dtype=np.uint8)

    for t in range(n):
        x = 0
        for panel, width in zip(panels, widths, strict=True):
            out[t, header : header + panel.shape[1], x : x + width] = panel[t]
            x += width + pad

    if labels is not None:
        if len(labels) != len(panels):
            raise ValueError(f"{len(labels)} labels for {len(panels)} sequences")
        font = ImageFont.load_default()
        for t in range(n):
            img = Image.fromarray(out[t])
            draw = ImageDraw.Draw(img)
            x = 0
            for label, width in zip(labels, widths, strict=True):
                draw.text((x + 3, 2), str(label), fill=(20, 20, 20), font=font)
                x += width + pad
            out[t] = np.asarray(img)
    return out


def save_gif(
    path: str | Path,
    frames: Array | Sequence[Array],
    *,
    fps: int = 8,
    scale: int = 1,
    labels: Sequence[str] | None = None,
    loop: int = 0,
    colors: int = 256,
    dither: bool = False,
) -> Path:
    """Write an animated GIF.

    Args:
        frames: one ``(T, C, H, W)`` sequence, or several to tile side by side.
        fps: playback rate.
        scale: integer upscale factor. Prefer rendering at the size you want --
            nearest-neighbour upscaling turns every source pixel into a block and
            cannot add detail. Useful only for genuinely tiny sources such as the
            32x32 sprite world.
        labels: per-panel captions, drawn above each tile.
        loop: ``0`` loops forever.
        colors: palette size, at most 256 (a GIF limit).
        dither: diffuse quantisation error. Off by default: on smooth renders it
            reads as grain, and with a 256-colour palette there is little error
            left to diffuse.

    A **single palette is computed across all frames**. Quantising each frame
    independently -- what Pillow does by default -- gives every frame its own
    palette, so colours shift frame to frame and the animation shimmers even
    when the underlying pixels barely change.
    """
    Image, _, _ = _pillow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, (list, tuple)):
        stack = tile_frames(frames, labels=labels, scale=scale)
    else:
        stack = upscale(frames_to_uint8(frames), scale)
        if labels is not None:
            stack = tile_frames([frames], labels=labels, scale=scale)

    images = [Image.fromarray(f) for f in stack]
    quantized = _quantize_shared(images, colors=colors, dither=dither)
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=max(int(round(1000 / fps)), 20),
        loop=loop,
        disposal=2,
        optimize=True,
    )
    return path


def _quantize_shared(images: list, *, colors: int, dither: bool):
    """Map every frame onto one palette derived from all of them."""
    Image, _, _ = _pillow()
    if not 2 <= colors <= 256:
        raise ValueError(f"colors must be in [2, 256], got {colors}")
    # Build the palette from every frame at once by stacking them vertically.
    montage = Image.new("RGB", (images[0].width, images[0].height * len(images)))
    for index, image in enumerate(images):
        montage.paste(image, (0, index * images[0].height))
    master = montage.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    mode = Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE
    return [image.quantize(palette=master, dither=mode) for image in images]


def save_figure(fig, path: str | Path, *, dpi: int = 150, close: bool = True) -> Path:
    """Save a matplotlib figure, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt

        plt.close(fig)
    return path
