# MuZero

**A latent model trained to agree with its own search.** MuZero does not try to
model the observation at all, neither its pixels nor its embedding. It learns
whatever latent makes three predictions come out right: reward, value, and policy.
And the policy target is not the behaviour policy; it is the *improved* policy that
tree search produced. The model is trained to agree with a better version of
itself.

```python
agent = xwm.families.muzero.muzero(n_actions=15, observation="state", state_dim=20)
```

## Three networks, in MuZero's own terms

| network | signature | attribute |
| --- | --- | --- |
| representation | observation → `s` | `encoder` |
| dynamics | `(s, a) → (s', r)` | `dynamics` |
| prediction | `s → (policy, value)` | `policy`, `value` |

```python
z = agent.represent(observation)      # representation
z_next, reward = agent.recurrent(z, action)
policy_logits, value = agent.predict(z)
```

## The loop that makes it MuZero

1. **Act by searching.** Run [MCTS](../reference/planning.md#xwm.planning.MCTS)
   through the current learned model.
2. **Store what the search concluded**, not what the policy did: the visit
   distribution as the policy target, the root value for the value bootstrap.
3. **Train** the three heads against those targets.
4. Repeat. Each round the model is trained to reproduce a search that ran through
   a *better* model than the one that generated the data.

```python
recurrent, predict = agent.search_fns()
mcts = xwm.planning.MCTS(n_actions=15, n_simulations=50, discount=0.997)

result = mcts.search(key, agent.represent(observation), recurrent, predict)

result.action    # visit-count argmax -- what to execute
result.policy    # the visit distribution -- the policy *target*
result.value     # root value -- feeds the n-step value target
```

## The loss

The dynamics unroll `horizon` steps from a real observation, and all three heads
are matched at every step against the stored targets:

| head | target | default weight |
| --- | --- | --- |
| reward | the observed reward | `reward_coef=1.0` |
| value | an n-step bootstrapped return | `value_coef=0.25` |
| policy | MCTS visit counts | `policy_coef=1.0` |

Because the unroll is recurrent and gradients flow through it, the gradient is
scaled by `1 / horizon`, which keeps its magnitude independent of the unroll
length.

Reward and value use symlog plus two-hot categorical prediction
([`CategoricalScalar`](../reference/heads.md#xwm.heads.CategoricalScalar)), the same
scale-free trick TD-MPC2 uses.

## Storing an episode

The value target is computed on the host, from the rewards and the search's own
root values:

```python
buffer = xwm.training.ReplayBuffer(
    capacity=200_000,
    observation_shape=(env.state_dim,),
    action_shape=(),                                    # a scalar action index
    extra={"value_target": (), "policy_target": (n_actions,)},
)

buffer.add_episode(
    observations, action_indices, rewards,
    value_target=xwm.families.muzero.n_step_value_targets(
        rewards, root_values, discount=0.997, n_steps=5,
    ),
    policy_target=visit_distributions,
)
```

[`n_step_value_targets`](../reference/families.md#xwm.families.muzero.n_step_value_targets)
computes \(\sum_{k<n} \gamma^k r_{t+k} + \gamma^n V_{\text{root}}(t+n)\), dropping
the bootstrap where the sum runs past the end of the episode.

!!! tip "Seed with random actions"

    Search through an untrained model is no better than noise and considerably
    more expensive. `examples/08_muzero_franka.py` seeds the buffer with random
    discrete actions and a uniform policy target before the first search episode.

## Discrete actions, and a 7-DoF arm

MuZero is a discrete-action algorithm, because that is what MCTS needs. To drive a
continuous arm, discretise:

```python
table = xwm.envs.discrete_action_table(action_dim=7, magnitude=1.0, include_noop=True)
# (15, 7) -- +/- on each joint, plus doing nothing

agent = xwm.families.muzero.muzero(n_actions=len(table), observation="state", state_dim=20)
env.step(table[result.action])
```

For continuous control proper, prefer [TD-MPC2](tdmpc2.md), whose MPPI planner is
native to continuous actions, or Sampled MuZero, which this module does not
implement.

## What we measured

!!! warning "A smoke run, not a result"

    The only MuZero numbers this library currently has are from the laptop
    defaults: 2 iterations, 8 simulations per move, a 65-step buffer. At that
    budget MuZero + MCTS ends up more than twice as far from the goal as doing
    nothing (0.505 m against a 0.214 m no-op baseline). That records a working
    pipeline, not a working agent, and it is not evidence about the algorithm.
    See [Findings](../findings.md#muzero-on-the-franka-arm).

The legible part is the **policy loss**: at 2.689 it is 99.3% of
\(\ln 15 = 2.708\), the cross-entropy of a head that outputs a uniform
distribution over 15 actions, so after two iterations the policy had learned
essentially nothing. The search entropy, by contrast, says nothing at all here:
with 8 simulations the root can visit at most 8 actions, so the statistic is
capped by the budget rather than shaped by the model.

MuZero is the most sample-hungry of the three families, and it needs the search to
be better than its own policy before its targets mean anything. If you want a
value-based agent on this task at a budget you can afford, start with
[TD-MPC2](tdmpc2.md).

## Example

`examples/08_muzero_franka.py` runs the full search-act-store-train loop on the
Franka arm. See [Examples](../guides/examples.md).
