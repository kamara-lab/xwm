# Installation

`xwm` needs Python ≥ 3.11. The core is pure JAX and installs anywhere JAX does.

```bash
pip install xwm
```

## From source

```bash
git clone https://github.com/kamara-lab/xwm
cd xwm
pip install -e ".[dev]"
```

## Extras

Each extra is additive and independent.

| extra | pulls in | when you need it |
| --- | --- | --- |
| *(core)* | `jax`, `equinox`, `optax`, `numpy`, `einops` | models, objectives, planners, training |
| `plots` | `matplotlib`, `pillow` | figures, GIFs, LaTeX/JSON tables |
| `newton` | `newton`, `warp-lang`, `GitPython`, `trimesh`, `pycollada`, `usd-core` | the Franka arm in [`xwm.envs`](../reference/envs.md) |
| `render` | `ovrtx`, `pyglet` | path-traced figures, NVIDIA GPUs only |
| `dev` | `pytest`, `ruff`, `matplotlib`, `pillow` | tests and linting |

```bash
pip install "xwm[plots]"            # figures and tables
pip install "xwm[newton]"           # the robot
pip install "xwm[newton,render]"    # the robot, path traced
```

!!! note "Why `newton` carries extra dependencies"

    Newton itself does not require them, but the Franka setup does: `GitPython`
    fetches the robot assets, `trimesh` and `pycollada` read the FR3 URDF's
    meshes, and `usd-core` is what lets `ViewerUSD` export a stage for offline
    rendering.

## GPU

Install the JAX wheel matching your accelerator *before* `xwm`, following
[JAX's own instructions](https://docs.jax.dev/en/latest/installation.html). The
CPU wheel is the default, and `xwm` does not override it.

```bash
pip install -U "jax[cuda12]"
pip install xwm
```

Nothing in the library is CUDA-specific. The Franka environment picks its
physics solver at run time: `mujoco_warp` where a CUDA GPU is available,
Featherstone otherwise.

## Checking the install

```python
import xwm

xwm.__version__
xwm.families.available()
# ['jepa/action', 'jepa/image', 'jepa/image-lejepa', 'jepa/video',
#  'jepa/video-lejepa', 'muzero', 'tdmpc2']
```

For the robot and the renderers:

```python
xwm.envs.which_backends()
# {'warp': True, 'rtx': False, 'usd': True}
```

`False` means the extra for that backend is missing or, for `rtx`, that no
graphics-capable NVIDIA device could be reached. See
[Rendering](../guides/rendering.md).

## Tests

```bash
pytest
```

The tests check *behaviour*, not just shapes: mask samplers must never leak a
target token into the context, dynamics must respond to their action input,
planners must reach a reachable goal, MCTS must find a payoff one step away,
frozen parameters must not move, and SIGReg must actually pull a skewed
distribution toward isotropy.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

The API reference imports `xwm`, so the package has to be installed, not merely
on the path, for `mkdocs build` to resolve signatures.

The world-model diagram is a TikZ picture, built separately and committed as SVG,
so neither the docs build nor CI needs a TeX installation. To change it, edit
`docs/assets/world-model.tex` and rebuild both themes:

```bash
cd docs/assets
pdflatex -jobname=world-model-light "\def\xwmtheme{light}\input{world-model.tex}"
pdflatex -jobname=world-model-dark  "\def\xwmtheme{dark}\input{world-model.tex}"
pdftocairo -svg world-model-light.pdf world-model-light.svg
pdftocairo -svg world-model-dark.pdf  world-model-dark.svg
```

The two variants are referenced with Material's `#only-light` and `#only-dark`
suffixes, so the diagram follows the site's theme toggle.
