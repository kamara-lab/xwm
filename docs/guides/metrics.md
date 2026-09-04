# Metrics

Every number `xwm` reports, what it means, and how it is computed. The
definitions matter more than usual here, because most of these quantities have a
plausible-looking variant that measures something else — and because two of
them, taken alone, are actively misleading.

Notation: $z$ is a latent state, $s$ a simulator state, $a$ an action, $H$ a
horizon, $T$ a number of frames, $B$ a batch, $D$ an embedding width.

---

## 1. Control metrics

Produced by `xwm.bench.evaluate`, one set per policy, over $N$ evaluation
instances. Each instance $i$ has a start state $s^i_0$, a goal state $g^i$ taken
from the same recording, and a trace of simulator states $s^i_1 \dots s^i_{T_i}$
under a step budget.

Two conventions apply throughout and are easy to get wrong:

- **Distances are in the environment's own units**, over *simulator* state —
  never over observations and never over latents. A success criterion computed
  from the quantity the model is optimising measures nothing.
- **The episode does not stop at success.** Arrival is latched, the episode runs
  to the budget, so a planner that reaches the goal and drifts off it is
  reported as such.

### Success rate

The headline. The fraction of instances in which the goal was reached *at any
point* within the budget.

$$
\text{success} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\exists\, t \le T_i : \ d(s^i_t, g^i) < \tau \right]
$$

$d$ and the threshold $\tau$ come from the task's metric (§3). Quoting this
without the budget, the goal offset and the `noop` floor is quoting nothing:
the same model scores anywhere from 0 to 1 depending on how far the goal was
placed.

### Distance closed

The fraction of the initial gap that was removed, averaged over instances.

$$
\text{closed}_i = 1 - \frac{d(s^i_{T_i}, g^i)}{d(s^i_0, g^i)}
\qquad
\text{closed} = \frac{1}{N}\sum_i \text{closed}_i
$$

$1$ is arrival; $0$ is no progress; **negative means the policy ended further
away than it started**, which is common and is why this is reported rather than
success alone. It is the metric that still discriminates when every policy
scores zero success — as they do early in training, where success rate has no
gradient of information at all.

Note the denominator is per instance, so an instance that begins close to its
goal cannot dominate the average. Instances that begin *already solved* are
excluded at sampling time (§5).

### Final, best and initial distance

$$
\text{final}_i = d(s^i_{T_i}, g^i)
\qquad
\text{best}_i = \min_{t \le T_i} d(s^i_t, g^i)
\qquad
\text{initial}_i = d(s^i_0, g^i)
$$

Reported as means, plus a median for `final` because a single diverging episode
moves a mean a long way.

**`best` under `final` is the signature of a planner that overshoots.** It got
there and could not stay, which is a different failure from never arriving and
calls for a different fix — usually a shorter receding horizon rather than a
better model.

### Steps to success

$$
\text{steps} = \operatorname*{mean}_{i \,:\, \text{solved}} \min\{t : d(s^i_t, g^i) < \tau\}
$$

Averaged over solved instances only, and reported beside `steps_to_success_n`,
the count it was averaged over. Without that count it is unreadable: a mean of
4 over one lucky instance and a mean of 4 over fifty are not the same claim.

### Plan cost

The planner's own estimate of the cost of the trajectory it chose — the value
of the objective in §4, in latent units. It is a **diagnostic, not a score**:
it is comparable across steps of one run and across seeds of one model, and it
is not comparable between models, because each model measures it in its own
latent space. A plan cost that falls while the true distance does not is the
clearest single sign that the model's dynamics disagree with the world.

### The floors

Three baselines run through the identical loop, on the same instances with the
same seeds:

| policy | what it does | reading |
| --- | --- | --- |
| `replay` | the demonstrator's recorded actions | **a gate.** Should be ~1.0. Below that, the environment is not being reset faithfully and no other number on this task means anything. |
| `noop` | zero action throughout | how much of the gap the start already closes. A high `noop` means the instances are trivial. |
| `random` | uniform in $[-1,1]^{kA}$ | the floor a search must beat to be searching. |

---

## 2. Prediction metrics

How well the model predicts, independently of whether it can plan. All are in
latent space, all over held-out episodes.

### Latent prediction loss

Given predicted latents $\hat z$ and targets $z$, `xwm.objectives.prediction_loss`
computes, for `kind` $\in$ {`l1`, `l2`, `smooth_l1`, `cosine`}:

$$
\mathcal{L}_{\ell_1} = \frac{1}{n}\sum \lvert \hat z - z \rvert
\qquad
\mathcal{L}_{\ell_2} = \frac{1}{n}\sum (\hat z - z)^2
$$

$$
\mathcal{L}_{\cos} = 1 - \frac{1}{n}\sum \frac{\langle \hat z, z\rangle}{\lVert \hat z\rVert\,\lVert z\rVert}
$$

### Teacher forcing versus rollout

The same loss under two regimes, and the gap between them is the point.

$$
\mathcal{L}_{\text{tf}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \ell\big(f(z_t, a_t),\ z_{t+1}\big)
$$

$$
\mathcal{L}_{\text{roll}} = \frac{1}{T-1}\sum_{t=1}^{T-1} \ell\big(\hat z_{t+1},\ z_{t+1}\big),
\quad \hat z_{t+1} = f(\hat z_t, a_t),\ \ \hat z_1 = z_1
$$

Teacher forcing steps from *ground-truth* latents: a dense, well-conditioned
signal that never shows the model its own mistakes. Rollout runs the whole
horizon from one latent with the model consuming its own predictions — the
regime a planner actually uses, and the only one that charges for compounding
error.

$\mathcal{L}_{\text{roll}} \ge \mathcal{L}_{\text{tf}}$ always, in practice. The
ratio is the compounding-error measure; a rollout loss that equals teacher
forcing means the horizon is 1 and the rollout term is telling you nothing.

### Horizon error and the identity floor

Rollout error as a function of horizon $h$, against the trivial predictor that
says nothing changes:

$$
e(h) = \ell\big(f^{(h)}(z_t, a_{t:t+h}),\ z_{t+h}\big)
\qquad
e_{\text{id}}(h) = \ell\big(z_t,\ z_{t+h}\big)
$$

**Always report $e_\text{id}$.** A small $e(h)$ means nothing on its own: in a
slow-moving environment, predicting no change is an excellent predictor, and a
model that has learned exactly that will show an impressive error curve.

---

## 3. Task metrics

The ground-truth distance and success predicate, defined per task over
*simulator* state.

### Sliced Euclidean (the default)

Most tasks care about part of the state. In a pushing task the goal is a puck
position and the pusher's own position is irrelevant — a planner that parks its
pusher where the demonstrator's was, with the puck untouched, has done nothing.

$$
d(s, g) = \lVert s_{[j:k]} - g_{[j:k]} \rVert_2
\qquad
\text{success} = d(s,g) < \tau
$$

| metric | slice | task |
| --- | --- | --- |
| `puck` | `state[2:4]` | `pusht/synthetic` — the pushed object, $\tau = 0.08$ |
| `agent` | `state[0:2]` | `maze/synthetic` — the agent's position |
| `euclidean` | all | the fallback |

### Push-T (the published criterion)

For the real Push-T task, position and angle are thresholded separately, and the
angle respects the block's symmetry:

$$
\lVert g_{[0:4]} - s_{[0:4]} \rVert_2 < 20\ \text{px}
\quad\text{and}\quad
\min\big(\lvert \Delta\theta \rvert \bmod \sigma,\ \sigma - (\lvert \Delta\theta \rvert \bmod \sigma)\big) < \pi/9
$$

with symmetry $\sigma = 2\pi$. This is the criterion behind every published
Push-T success rate, which is why it is used in preference to `gym-pusht`'s own
coverage flag — that one is measured against a *fixed* target the
dataset-driven protocol has replaced.

---

## 4. Planning objectives

What the planner minimises. Not a metric — it is what the metrics judge — but
it belongs here because it is measured in the same latent units as plan cost.

### Goal cost

$$
C(\hat z_{1:H}, a_{1:H}) = \sum_{t=1}^{H} \gamma^{t}\, \delta\big(\rho(\hat z_t),\, z_g\big) \;+\; \lambda \lVert a_t \rVert^2
$$

$\delta$ is `l1`, `l2` or `cosine`; $z_g$ is the *encoded goal observation*;
$\rho$ is the model's `readout`, which selects the newest frame from a latent
that carries several, so stale context is not charged for failing to move. With
`terminal_only`, only $t = H$ contributes.

Every model supports this objective, which is what makes families comparable:
a reward-free JEPA and a TD-MPC2 are asked the same question.

### Return cost

$$
C = -\left( \sum_{t=1}^{H} \gamma^{t}\, r(\hat z_t, a_t) \;+\; \gamma^{H+1} V(\hat z_H) \right)
$$

The terminal value is the whole point: it summarises everything past the
horizon, which is why a horizon-3 planner can act as though it saw further.
Only available to a family with reward and value heads, and it must be scored at
the *pre-transition* latent because that is how $r$ was trained — so this
objective is built by the family that owns the heads, not assembled from
outside.

---

## 5. Sampling and instance validity

An instance is drawn from a held-out recording: an episode $i$, a start frame
$t$, and a goal frame $t + \Delta$ where $\Delta = \lfloor \text{goal\_offset} /
k \rfloor$ for frameskip $k$. Candidates are filtered:

$$
d\big(s^{(i)}_t,\ s^{(i)}_{t+\Delta}\big) \ge \tau
$$

An instance failing this is one the `noop` baseline solves by definition.
Drawing them flatters every policy equally and measures none of them — a
failure mode that is not hypothetical: the synthetic pushing task initially
scored `noop` and `random` at 100% for exactly this reason (see
[findings](../findings.md)).

Sampling is seeded and model-independent, so two models see literally the same
instances and their numbers are paired.

---

## 6. Representation diagnostics

From `xwm.metrics`, over a batch of embeddings $Z \in \mathbb{R}^{n \times D}$.

**The loss is not the metric.** A collapsing encoder drives its prediction loss
*down* — it is predicting its own degenerate output — so these are the numbers
that say whether a falling loss is progress or collapse.

### RankMe

The effective rank of the embedding spectrum, via the entropy of the normalised
singular values $\sigma_1 \dots \sigma_D$ of $Z$:

$$
p_i = \frac{\sigma_i}{\sum_j \sigma_j} + \varepsilon
\qquad
\text{RankMe}(Z) = \exp\left(-\sum_{i=1}^{D} p_i \log p_i\right)
$$

Ranges from $1$ (all variance in one direction) to $D$ (isotropic). Computed
after centring, which is exactly why it cannot be read alone: **a constant
offset is invisible to it.**

### Feature standard deviation

$$
\text{feature\_std}(Z) = \frac{1}{D}\sum_{j=1}^{D} \operatorname{std}(Z_{:,j})
$$

$\to 0$ is collapse, and unlike RankMe it does see a constant.

### Mean cosine similarity

$$
\text{mean\_cos}(Z) = \frac{2}{n(n-1)}\sum_{i<j} \frac{\langle z_i, z_j \rangle}{\lVert z_i \rVert \lVert z_j \rVert}
$$

$\to 1$ is collapse. It catches the case the other two miss: embeddings with
healthy variance and high effective rank that nonetheless share a large common
component. A randomly-initialised frozen ViT scores RankMe 19.5 and
mean cosine 0.999 on this library's synthetic frames — high rank, nearly
collinear — and a goal cost through such an encoder has almost nothing to
discriminate on.

Report all three. Each misses a case the others catch, which is why
`collapse_report` returns them together.

### Probes

`ridge_probe` and `knn_probe` measure how much of a known quantity a
representation retains — $R^2$ for regression, accuracy for classification.

A probe is **not** a collapse detector. `ridge_probe` standardises its features,
so it amplifies a nearly-dead signal back to full scale and can report a
respectable $R^2$ on a representation that has all but collapsed.

---

## 7. Regularizers

Not metrics, but they are logged as losses and read as diagnostics.

### SIGReg

The LeJEPA objective: the embedding distribution should be an isotropic
Gaussian. Testing that in $D$ dimensions directly is expensive, so it is
sketched — for $z \sim \mathcal{N}(0, I_D)$ and any unit $v$, the projection
$\langle z, v\rangle$ is exactly $\mathcal{N}(0,1)$ regardless of $D$:

$$
\text{SIGReg}(Z) = \frac{1}{P}\sum_{p=1}^{P} S\big(Z v_p\big),
\qquad v_p \sim \mathrm{Unif}(\mathbb{S}^{D-1})
$$

with $S$ a one-dimensional goodness-of-fit statistic. The default is
Epps–Pulley, comparing empirical and target characteristic functions on a
quadrature grid:

$$
S(u) = \int \left( \Big\lvert \tfrac{1}{n}\sum_m e^{\mathrm{i} t u_m} - e^{-t^2/2} \Big\rvert^2 \right) w(t)\, \mathrm{d}t
$$

**The quadrature convention changes the scale by a constant, and published
weights assume a specific one.** xwm's default normalises the weights and does
not scale by $n$; the LeJEPA-family models integrate $e^{-t^2/2}$ unnormalised
over 17 knots and multiply by $n$, giving

$$
S_{\text{ref}} \approx n\sqrt{2\pi}\; S_{\text{xwm}}
$$

— a factor of ~320 at batch 128. Pass `quadrature="trapezoid", n_nodes=17,
scale_by_n=True` to make a paper's `reg_weight` mean what it says. The `axis`
argument selects per-timestep testing over pooled: pooling asks whether the
*union* over time is isotropic, which a mixture of a too-narrow and a too-wide
slice can satisfy while neither slice does.

### VICReg

Variance, invariance and covariance terms:

$$
\mathcal{L}_{\text{var}} = \frac{1}{D}\sum_j \max\big(0,\ \gamma - \operatorname{std}(Z_{:,j})\big)
\qquad
\mathcal{L}_{\text{cov}} = \frac{1}{D}\sum_{j \ne k} \operatorname{Cov}(Z)_{jk}^2
$$

The variance hinge forbids collapse directly; the covariance term decorrelates
the dimensions so they carry different information.

---

## Where these appear

| file | contents |
| --- | --- |
| `eval.json` | §1 per policy, plus the instance list and the full task spec |
| `eval_table.tex` | §1 as a paper-ready table |
| `history.json` | §2 and §7 per logged training step |
| `collapse_report(z)` | §6, all three together |

## References

- Zhou et al., *DINO-WM*, 2024. [arXiv:2411.04983](https://arxiv.org/abs/2411.04983)
- Maes et al., *stable-worldmodel*, 2026. [arXiv:2605.21800](https://arxiv.org/abs/2605.21800)
- Balestriero & LeCun, *LeJEPA*, 2025 — SIGReg
- Garrido et al., *RankMe*, ICML 2023. [arXiv:2210.02885](https://arxiv.org/abs/2210.02885)
- Bardes, Ponce & LeCun, *VICReg*, ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- Epps & Pulley, *A Test for Normality Based on the Empirical Characteristic Function*, Biometrika 1983
- Chi et al., *Diffusion Policy*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) — the Push-T criterion
