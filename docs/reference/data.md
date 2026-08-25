# xwm.data

Batch streams and three synthetic worlds, all pure JAX and all fast enough to train and plan on CPU. `SpriteWorld` is a sprite the actions move plus distractors that move on their own, optionally behind occluders. `PushWorld` adds contact: a puck that moves only when pushed, with a dense reward. `MazeWorld` adds walls and distance: a sparse goal several corridors away.

For datasets someone else recorded, see [`xwm.datasets`](datasets.md).

::: xwm.data
