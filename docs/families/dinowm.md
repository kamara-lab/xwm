# DINO-WM

**Dynamics on frozen DINOv2 patch features.** The encoder is not trained here at
all, and never saw a robot.

```python
model = xwm.families.dinowm.dinowm(action_dim=7, img_size=224)
```

That makes it the **control** for every model that does learn a representation.
If a general visual encoder, trained on internet images with no action labels,
plans as well as one fitted to the task, then the representation learning was
not what was doing the work. It is the cheapest way to find out, because there
is nothing to pretrain.

## Components

| component | signature | class |
| --- | --- | --- |
| encoder | `(3, H, W) → (N, 384)` | [`DINOv2Encoder`](../reference/encoders.md#xwm.encoders.DINOv2Encoder), frozen |
| dynamics | `(H, N, D) × (H, A) → (H, N, D)` | [`ARPredictor`](../reference/dynamics.md#xwm.dynamics.ARPredictor), `conditioning="concat"` |

It is [`ARWorldModel`](lewm.md) with `tokens="patch"` and `freeze_encoder=True`.
It is its own family because what it claims is its own, not because the code is.

## Patches, not a pooled vector

The other windowed models pool each frame to a single vector. DINO-WM must not:
pooling DINOv2's patches averages away *where* things are, which is most of what
a manipulation task is about. So the latent is the whole grid, and the action is
**concatenated to every token** rather than modulating the frame as a whole.

The price is tokens, and it is not small.

| model | tokens per frame at 224px | window of 3 |
| --- | --- | --- |
| `lewm`, `delta_jepa` | 1 | 3 |
| `dinowm` | 256 | 768 |

That ratio is why LeWorldModel plans faster, and it is measured here rather than
quoted. On the synthetic pusher, same instances, same budget, same planner:
**532 seconds against 55**, at a resolution where DINO-WM's window is 48 tokens
and the pooled models' is 3. `docs/findings.md` has the table.

## Loading the weights

```bash
pip install "xwm[pretrained]"
```

```python
encoder = xwm.encoders.dinov2("small", img_size=224)   # or base, large, giant
encoder(image).shape                                    # (256, 384)
```

There is no PyTorch in this path and nothing calls out to another runtime.
[`read_safetensors`](../reference/tools.md#xwm.tools.read_safetensors) reads the
published checkpoint as numpy and the tensors are placed into Equinox modules.

!!! note "Two things that are 'nearly right' and therefore wrong"

    A conversion bug in a vision transformer does not raise. It produces
    finite, normal-looking features that are a different encoder from the one
    everyone else compares against. Two were found by checking against the
    reference implementation rather than by inspection:

    * **Bicubic is not one filter.** `jax.image.resize(method="bicubic")` uses
      Keys' cubic convolution with `a = -0.5`; PyTorch uses `a = -0.75`. The
      position table is resampled from its native 37×37 grid to whatever
      resolution you ask for, so this is on the path for every input size
      except 518. Worth 0.007 of cosine similarity in the output.
    * **GELU is not one function.** `jax.nn.gelu` defaults to the tanh
      approximation; DINOv2 was trained with PyTorch's exact one.

    With both corrected the encoder matches the reference to float32 noise
    (cosine `0.99999994`). The test that asserts this is gated behind
    `XWM_PRETRAINED_TESTS=1` because it downloads.

## Resolution

DINOv2 has 14-pixel patches, so `img_size` must be a multiple of 14 and the
model says so with the nearest one if it is not. The position table is
*resampled* onto the resulting grid, never truncated — keeping the first `gh·gw`
rows would place every patch in the top-left corner of the picture.

## No proprioception

The paper conditions on joint positions on some benchmarks. That is not
implemented here, and it is not a flag. A planner proposes actions, not future
joint angles, so a proprioceptive input has to be *predicted* as part of the
latent rather than supplied — a real extension rather than an argument.

## References

| model | paper |
| --- | --- |
| DINO-WM | Zhou et al., ICML 2025 · [arXiv:2411.04983](https://arxiv.org/abs/2411.04983) |
| DINOv2 | Oquab et al., TMLR 2024 · [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| registers | Darcet et al., ICLR 2024 · [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) |
