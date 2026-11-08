# sim: simulated audience

The primary RL evaluation for M1 (design.md, "Simulated audience environment"). Plain numpy,
gym-style `reset`/`step`, thousands of episodes per second.

```python
from sim.env import AudienceEnv, Context
env = AudienceEnv(shift_at=20)                     # optional mid-set preference shift
obs = env.reset(Context("bar", "normal", "mixed"), seed=0)
obs, reward, done, info = env.step(1)              # mode for the next bit (index into env.modes)
info["levels"], info["regret"]                     # 0-3 level per joke in the bit; oracle gap
obs["last_mode"], obs["last_reward"]               # history for context-aware policies
```

| Piece | Model |
|---|---|
| Decision | One mode per **bit** (8 bits × 5 jokes); the mode is fixed within a bit |
| Latent (per joke) | `base[mode] + venue[mode] + age[mode] + energy + novelty·e^(−joke_t/τ) − fatigue·uses_in_last_2_bits + shift` |
| Laugh level | latent + N(0, noise), counted against thresholds `[0, 1, 2]` → 0–3 |
| Reward | mean level of the bit's jokes / 3 (samplers work on [0, 1]; logs keep the levels) |
| Oracle | `expected_rewards()`: exact E[mean level]/3 per mode for the next bit; regret = best − chosen |
| Planning optimum | `evaluate_planning.PlanningOracle`: exact best sequence for the show, by dynamic programming; the true ceiling in steady shows |
| ε-soft director | `policies.EpsilonSoft`: wraps a policy, explores uniformly with probability ε, exact `probs(obs)`; cost measured by `evaluate_epsilon` (Exp 18) |

All numbers live in [audience.yaml](audience.yaml) (the room's hidden truth, not the
director's prior). Only `base` has a source (HumorGen judge win
rates); venue/age/energy offsets are made up and the report must say so. PR 3 swaps `base`
for rJokes-derived values.

Not modeled yet: callbacks (delayed reward), tag-or-move-on within a bit. No switching penalty:
per-bit decisions already keep the voice coherent (decisions.md 2026-10-27).

Metrics: `python -m sim.evaluate_env` → [docs/metrics.md](../docs/metrics.md).

## Policies

[policies.py](policies.py): `Fixed`, `Random`, `EpsilonGreedy` (per-context table), `LinUCB` and
`LinTS` (one ridge model per mode over context + last-bit features; prior = starting bias weight).
Chosen director: **LinUCB + uniform prior** (decisions.md 2026-10-27).
Compare all policies × priors: `python -m sim.evaluate_policies` → docs/metrics.md.
