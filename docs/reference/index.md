# API reference

Fifteen modules, in the order they compose. Each page renders the module's own
docstring, including the `References` block naming the paper the code follows, and
then every public symbol.

<div class="grid cards" markdown>

-   __Foundations__

    ---

    [`xwm.core`](core.md) · types, base modules, EMA, rollouts, keys<br>
    [`xwm.nn`](nn.md) · attention, transformers, RoPE, patches<br>
    [`xwm.encoders`](encoders.md) · observation → latent<br>
    [`xwm.dynamics`](dynamics.md) · **`(z, a) → z'`**<br>
    [`xwm.heads`](heads.md) · reward, value, policy, Q-ensemble

-   __Training signals__

    ---

    [`xwm.masking`](masking.md) · blocks, tubes, temporal splits<br>
    [`xwm.objectives`](objectives.md) · prediction, SIGReg, VICReg, InfoNCE<br>
    [`xwm.families`](families.md) · `jepa`, `tdmpc2`, `muzero`<br>
    [`xwm.training`](training.md) · `Trainer`, schedules, replay

-   __Acting__

    ---

    [`xwm.planning`](planning.md) · CEM, MPPI, gradient, MPC, MCTS<br>
    [`xwm.envs`](envs.md) · a Franka FR3 in Newton<br>
    [`xwm.data`](data.md) · batch streams, a synthetic world

-   __Measuring__

    ---

    [`xwm.metrics`](metrics.md) · probes and collapse diagnostics<br>
    [`xwm.plots`](plots.md) · figures, GIFs, tables<br>
    [`xwm.tools`](tools.md) · checkpoints, summaries

</div>

## Top level

The package re-exports the handful of names most code needs:

::: xwm
    options:
      members:
        - set_seed
        - seed
        - key_source
        - Module
        - WorldModel
      show_root_heading: false
      summary: false

## Citations

Every module carries a `References` block in its docstring naming the paper the
code follows, so the citation sits beside the implementation:

```python
help(xwm.families.tdmpc2.model)
```

| model | paper |
| --- | --- |
| I-JEPA | Assran et al., CVPR 2023 · [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) |
| V-JEPA | Bardes et al., 2024 · [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) |
| V-JEPA 2 / -AC | Assran et al., *V-JEPA 2*, 2025 |
| LeJEPA | Balestriero & LeCun, 2025 |
| TD-MPC2 | Hansen, Su & Wang, ICLR 2024 · [arXiv:2310.16828](https://arxiv.org/abs/2310.16828) |
| TD-MPC | Hansen, Wang & Su, ICML 2022 · [arXiv:2203.04955](https://arxiv.org/abs/2203.04955) |
| MuZero | Schrittwieser et al., Nature 2020 · [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) |
| Sampled MuZero | Hubert et al., ICML 2021 · [arXiv:2104.06303](https://arxiv.org/abs/2104.06303) |
| VICReg | Bardes, Ponce & LeCun, ICLR 2022 · [arXiv:2105.04906](https://arxiv.org/abs/2105.04906) |

Component-level citations live in the docstrings of the modules that implement
them: SimNorm, two-hot categorical scalars, REDQ, SAC, MPPI, PUCT, Epps–Pulley,
RankMe, ViT/ViViT, MAE, RoPE, LayerScale, Mish.
