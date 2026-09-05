"""Latents and plans: what the model imagined, and what it proposed.

The two things a static figure cannot show about a world model. A rollout is a
path through a space of hundreds of dimensions, and the interesting question --
*where* does the imagined path leave the real one -- is a shape, not a number.
:func:`log_latent_trajectory` projects both paths through the *same* basis, so
they are comparable, and logs the per-horizon error beside them, because a
projection to three dimensions can hide error in the discarded ones.
"""

from __future__ import annotations

import numpy as np

from ._backend import rerun

__all__ = ["log_clip", "log_latent_trajectory", "log_plan"]


def _pca3(reference: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Project every argument onto the top three right-singular vectors of ``reference``."""
    flat = reference.reshape(reference.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    basis = vt[:3].T
    if basis.shape[1] < 3:  # a degenerate reference: pad so the viewer still gets 3-D
        basis = np.pad(basis, ((0, 0), (0, 3 - basis.shape[1])))
    out = [(flat - mean) @ basis]
    for other in others:
        other_flat = other.reshape(other.shape[0], -1)
        out.append((other_flat - mean) @ basis)
    return [np.asarray(x, np.float32) for x in out]


def log_clip(recording, frames, *, name: str = "clip", timeline: str = "frame") -> None:
    """Log a ``(T, C, H, W)`` clip frame by frame on its own timeline.

    Reuses :func:`xwm.plots.frames_to_uint8`, so channel order and dtype are
    decided in one place for figures and for Rerun alike.
    """
    if recording is None:
        return
    from ..plots.export import frames_to_uint8

    rr = rerun()
    images = frames_to_uint8(np.asarray(frames))
    for t, image in enumerate(images):
        recording.set_time(timeline, t)
        recording.log(f"clip/{name}", rr.Image(image))


def log_latent_trajectory(
    recording,
    true_z,
    imagined_z,
    *,
    prefix: str = "latent",
    timeline: str = "horizon",
) -> None:
    """Compare a real latent trajectory with an imagined one.

    Args:
        true_z: ``(H, ...)`` latents from encoding the observations that
            actually occurred.
        imagined_z: ``(H, ...)`` latents the dynamics produced from the first
            one, e.g. from :func:`xwm.core.rollout`.

    Logs both paths as 3-D line strips in a shared PCA basis fitted on
    ``true_z``, the imagined points on the ``horizon`` timeline, and
    ``<prefix>/error`` -- the full-dimensional L2 distance per step, which is
    the number that actually says how far the imagination drifted.
    """
    if recording is None:
        return
    rr = rerun()
    true_np = np.asarray(true_z, np.float32)
    imagined_np = np.asarray(imagined_z, np.float32)
    if true_np.shape != imagined_np.shape:
        raise ValueError(
            f"true and imagined trajectories differ: {true_np.shape} vs {imagined_np.shape}"
        )
    true_3d, imagined_3d = _pca3(true_np, imagined_np)
    recording.log(
        f"{prefix}/true", rr.LineStrips3D([true_3d], colors=[[80, 160, 255]]), static=True
    )
    recording.log(
        f"{prefix}/imagined", rr.LineStrips3D([imagined_3d], colors=[[255, 140, 60]]), static=True
    )

    flat_true = true_np.reshape(true_np.shape[0], -1)
    flat_imagined = imagined_np.reshape(imagined_np.shape[0], -1)
    error = np.linalg.norm(flat_imagined - flat_true, axis=-1)
    for t in range(true_np.shape[0]):
        recording.set_time(timeline, t)
        recording.log(f"{prefix}/point/true", rr.Points3D([true_3d[t]], colors=[[80, 160, 255]]))
        recording.log(
            f"{prefix}/point/imagined", rr.Points3D([imagined_3d[t]], colors=[[255, 140, 60]])
        )
        recording.log(f"{prefix}/error", rr.Scalars(float(error[t])))


def log_plan(recording, plan, *, prefix: str = "plan", timeline: str = "horizon") -> None:
    """Log a planner's :class:`~xwm.planning.Plan`: its proposal and its spread.

    ``mean`` and ``std`` are the sampling distribution the search converged to,
    so a plan whose ``std`` never shrinks is one the planner never became
    confident about -- visible here, invisible in the executed action alone.
    The candidate sets themselves are consumed inside the planner's
    ``lax.fori_loop`` and are not available to log.
    """
    if recording is None:
        return
    rr = rerun()
    mean = np.asarray(plan.mean, np.float32)
    std = np.asarray(plan.std, np.float32)
    actions = np.asarray(plan.actions, np.float32)
    recording.log(f"{prefix}/cost", rr.Scalars(float(plan.cost)))
    for t in range(mean.shape[0]):
        recording.set_time(timeline, t)
        for i in range(mean.shape[1]):
            recording.log(f"{prefix}/mean/{i}", rr.Scalars(float(mean[t, i])))
            recording.log(f"{prefix}/std/{i}", rr.Scalars(float(std[t, i])))
            recording.log(f"{prefix}/action/{i}", rr.Scalars(float(actions[t, i])))
