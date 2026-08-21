# What a JEPA predicts

A mask sampler splits the token grid into a **visible context** and one or more
**target blocks**. The context encoder computes only the visible tokens, which is
where the speedup over reconstruction comes from: masked positions cost nothing,
because they are never in the sequence at all.

<figure markdown="span">
  ![Multi-block masks drawn on a 2-D token grid](../outputs/01_image_ijepa/masks.png){ width="620" }
  <figcaption>
    Four independent draws of <code>MultiBlockMask2d</code> over one 2-D grid.
    Each draw yields one context and several target blocks.
  </figcaption>
</figure>

## The samplers

| sampler | used by | idea |
| --- | --- | --- |
| [`MultiBlockMask2d`](../reference/masking.md#xwm.masking.MultiBlockMask2d) | I-JEPA | large 2-D blocks, too big to interpolate from neighbours |
| [`TubeMask3d`](../reference/masking.md#xwm.masking.TubeMask3d) | V-JEPA | a spatial region extended through time, so no visible frame contains the answer |
| [`TemporalSplit`](../reference/masking.md#xwm.masking.TemporalSplit) | V-JEPA 2-AC | see a prefix, predict whole future frames |
| [`RandomMask`](../reference/masking.md#xwm.masking.RandomMask) | baselines | uniform random tokens |

Every sampler returns a [`MaskBatch`](../reference/masking.md#xwm.masking.MaskBatch):

```python
mask.context   # (K_ctx,)     int32 indices the encoder may see
mask.targets   # (M, K_tgt)   int32 indices to predict, one row per block
```

Several target blocks per draw let one context encoding be reused for several
prediction problems, which is where most of I-JEPA's efficiency comes from.

## Why the block has to be big

A small mask is not a prediction problem: a 4×4 hole in a natural image is
recoverable by interpolating its neighbours, and a model that learns to
interpolate has learned a low-level smoothness prior rather than anything about
the scene. `target_scale=0.15`, which is 15% of the grid per block, is large enough that
local continuation does not suffice.

The same argument in time is why `TubeMask3d` extends a spatial region *through*
frames rather than masking pixels per frame. If a region is visible in frame 3, a
model can copy it into masked frame 4 and score well without modelling motion at
all. A tube removes that shortcut: no visible frame contains the answer.

<figure markdown="span">
  ![Tube masks over a clip](../outputs/02_video_vjepa/tube_mask_frames.png){ width="620" }
  <figcaption>
    <code>TubeMask3d</code>: one spatial region, held across every frame of the clip.
  </figcaption>
</figure>

## Short- versus long-range

Two presets trade the number of prediction problems against their difficulty:

```python
sampler = xwm.masking.short_range_tubes(grid)   # many small tubes
sampler = xwm.masking.long_range_tubes(grid)    # few large tubes
```

Measured on 128×128×8 clips over 4000 steps
([findings](../findings.md#video-tube-masking)):

| preset | tubes | tokens/tube | context | loss | rankme |
| --- | --- | --- | --- | --- | --- |
| short-range | 8 | 36 | 69 | 0.451 | 15.1 |
| long-range | 2 | 100 | 86 | 0.273 | 22.7 |

The long-range preset has the *lower* loss and the *higher* effective rank. Fewer,
larger tubes leave a larger context, so each individual prediction is better
supported, and the representation that supports it is richer.

## Masks are batch-shared and statically shaped

Two properties that matter for compilation:

- **Batch-shared.** One draw applies to the whole batch. Per-sample masks would
  give each element a different sequence length.
- **Statically shaped.** Block sizes are fixed at construction, so `context` and
  `targets` have the same shape on every step and a training step compiles once.

Sampling is combinatorial host-side work (rejection sampling for non-overlapping
blocks), so it happens in `model.prepare_batch()`, outside `jit`. The
[`Trainer`](../reference/training.md#xwm.training.Trainer) calls it for you:

```python
batch = model.prepare_batch(batch, key)   # adds the mask; the Trainer does this
```

!!! warning "Never let a target token into the context"

    The tests assert this for every sampler, because it is the one bug that makes
    a JEPA look excellent and teach nothing: if the answer is visible, the
    predictor learns to copy.

## Sampling a mask directly

```python
import jax.random as jr
import xwm

grid = (8, 8)
sampler = xwm.masking.MultiBlockMask2d(grid, n_targets=4, target_scale=0.15)
mask = sampler(jr.PRNGKey(0))

z_context = model.encoder(image, keep=mask.context)     # only 85% of the grid
z_target = xwm.masking.gather(all_tokens, mask.targets[0])
```

[`complement`](../reference/masking.md#xwm.masking.complement),
[`subsample`](../reference/masking.md#xwm.masking.subsample) and
[`frame_indices`](../reference/masking.md#xwm.masking.frame_indices) cover the
index arithmetic you would otherwise write yourself.
