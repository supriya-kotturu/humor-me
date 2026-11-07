#!/usr/bin/env python
# coding: utf-8

# # humor-me: decisions and the experiments behind them
# 
# **One place for every decision.** The written log is [`docs/decisions.md`](../docs/decisions.md);
# this notebook is its **evidence**: each decision that rests on an experiment has a section here
# that re-runs it, shows the table/figure, and **checks the claim** the decision depends on.
# 
# **How to use it**
# - Open it and read: results load from `reports/metrics/*.json` (cached by parameters), so
#   nothing recomputes unless you ask.
# - Change a parameter in a cell → that experiment recomputes (seconds to ~3 min) and the cache updates.
# - `RERUN = True` below recomputes everything (~6 min).
# - Every claim has a **✅ / ❌ check** under it. If a code or parameter change flips one to ❌,
#   the decision it backs needs revisiting.
# - Before committing: `python scripts/run_notebook.py` (restart + run all, clean metadata).
# 
# Rules: logic lives in `sim/` and `pipeline/` functions; cells only call, display and check.
# Vocabulary: [glossary](../docs/wiki/glossary.md).

# In[1]:


import os, re, sys, json
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from IPython.display import Markdown, display
from sim import evaluate_env, evaluate_policies, evaluate_shift
from sim.cache import load, run_or_load

RERUN = False  # True = recompute every experiment (~6 min)

def show(md_text):
    display(Markdown(md_text))

def check(claim, ok):
    show(f"{'✅' if ok else '❌'} **{claim}**")
    return ok


# ## Index: every decision
# 
# Parsed live from `docs/decisions.md`, so it can't go stale. **Evidence** points at the section
# below that re-runs it; "rationale" means the decision rests on reasoning or constraints, not an
# experiment (the reasoning is in the log).

# In[2]:


EVIDENCE = {  # title fragment in decisions.md -> section in this notebook
    "S4: Gemini directly on whole sets": "Exp 1 Bit segmentation",
    "Bit = smallest standalone unit": "Exp 1 Bit segmentation",
    "Pause the transcript pipeline": "Exp 2 rJokes prep",
    "picks a mode per bit": "Exp 3 Per-joke vs per-bit",
    "Director policy: LinUCB": "Exp 4 Policies × priors",
    "Keep the director's features": "Exp 5 Mid-set shift",
    "Keep 8-bit sets": "Exp 6 Set length vs shift",
    "Show state step A": "Exp 7 Show state A",
}
rows = []
for date, title in re.findall(r"^## (\d{4}-\d{2}-\d{2}) — (.+)$", Path("docs/decisions.md").read_text(encoding="utf-8"), re.M):
    ev = next((v for k, v in EVIDENCE.items() if k in title), "rationale")
    rows.append(f"| {len(rows) + 1} | {date} | {title} | {ev} |")
show("| # | Date | Decision | Evidence |\n|---|---|---|---|\n" + "\n".join(rows))


# ## Exp 1 Bit segmentation on stand-up transcripts (2026-10-21)
# 
# **Decision:** S4 asks Gemini for bit boundaries on whole sets; a bit = smallest standalone stretch.
# **Where it fits:** the transcript pipeline that gives the report its "how a comedian builds a set"
# evidence ([stand-up structure](../docs/wiki/standup-structure.md)). Paused since 2026-10-26.
# **Question:** does an LLM find bit boundaries well enough, against hand labels and an
# evenly-spaced baseline?
# 
# **How to read:** F1 = boundary matches (higher better); Pk / WindowDiff = segmentation error
# (lower better). *Not re-runnable here*: needs Gemini + the local transcripts
# (`python -m pipeline.bits`, then `python -m pipeline.validate`); this is the stored snapshot.

# In[3]:


seg = load("segmentation_bargatze", {"source": "python -m pipeline.validate (S6)", "set": "nate-bargatze_hello-world_2023"})["sets"][0]
rows = []
for level in ("bits", "threads"):
    for who in ("llm", "even_baseline"):
        m = seg[level][who]
        rows.append(f"| {level} | {who} | {m['f1']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['pk']:.3f} | {m['windowdiff']:.3f} |")
show("| Level | Segmenter | F1 | P | R | Pk | WindowDiff |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows)
     + f"\n\nCallbacks: {seg['callbacks']['hyp']} found of {seg['callbacks']['ref']} hand-labeled.")
check("LLM bit boundaries beat the even-spacing baseline by > 0.5 F1",
      seg["bits"]["llm"]["f1"] - seg["bits"]["even_baseline"]["f1"] > 0.5)
check("Callbacks are not found yet (why they need their own pass)", seg["callbacks"]["hyp"] == 0)


# WindowDiff (Pevzner & Hearst, 2002) is:
# 
# $$\text{WindowDiff}(\text{ref}, \text{hyp}) = \frac{1}{N-k} \sum_{i=1}^{N-k} \mathbb{1}\Big[\, b_{\text{ref}}(i, i+k) \neq b_{\text{hyp}}(i, i+k) \,\Big]$$
# $N$ is the number of units in the set (here, probably lines or sentences of the transcript).
# $k$ is the window width, usually set to half the average segment length in the reference.
# $b(i, i+k)$ is the number of boundaries between unit $i$ and unit $i+k$.

# ## Exp 2 rJokes prep (2026-10-26: pause transcripts, rJokes + simulator first)
# 
# **Decision:** stop spending Gemini quota on transcripts; use rJokes as the working joke dataset.
# **Where it fits:** sim priors and the human-preference work ([rJokes](../docs/wiki/rjokes-dataset.md)).
# **How to read:** row counts through each cleaning step; repost spread is the **noise ceiling**
# (the same joke text scoring very differently). *Needs the local dataset to re-run*
# (`python -m pipeline.rjokes`); stored snapshot shown.

# In[4]:


rj = load("rjokes_prep", {"source": "python -m pipeline.rjokes (R1)", "min_words": 4, "max_words": 200})
show(f"""| Step | Jokes |
|---|---|
| Official splits | {rj['rows_in']:,} |
| After deduping reposts | {rj['after_dedupe']:,} |
| After 4–200 word filter | {rj['after_length']:,} |

Reposted texts: {rj['reposted']:,}; median score spread between posts of the same text:
**{rj['repost_spread_median']}** on the ln scale (≈ 7× upvotes).""")
check("Enough jokes remain for a 30k labeled sample (> 300k)", rj["after_length"] > 300_000)
check("Repost noise is large (median spread ≥ 2 on the ln scale)", rj["repost_spread_median"] >= 2)


# In[5]:


per_joke = run_or_load("sim_env_per_joke", evaluate_env.run, rerun=RERUN, episodes=1000, n_bits=40, jokes_per_bit=1, fatigue_window=5)
per_bit = run_or_load("sim_env", evaluate_env.run, rerun=RERUN, episodes=1000, n_bits=8, jokes_per_bit=5, fatigue_window=2)
show("**Per joke** (40 × 1)\n\n" + evaluate_env.table(per_joke))
show("**Per bit** (8 × 5)\n\n" + evaluate_env.table(per_bit))


# In[6]:


check("Per joke: random beats the best fixed mode in every venue (sim rewards rotation)",
      all(r["random"] > r["best_fixed"] for r in per_joke["venues"]))
check("Per bit: the best fixed mode beats random in every venue (fit to the room matters)",
      all(r["best_fixed"] > r["random"] for r in per_bit["venues"]))
check("Per bit: the oracle still beats the best fixed mode (adapting has headroom)",
      all(r["oracle"] > r["best_fixed"] + 2 for r in per_bit["venues"]))
check("The best mode differs by venue", len({r["best_mode"] for r in per_bit["venues"]}) == 3)


# ## Exp 4 Director policy × prior (2026-10-27)
# 
# **Decision:** LinUCB with a **uniform prior**.
# **Where it fits:** the director's learning rule and its starting guess
# ([glossary: prior](../docs/wiki/glossary.md#learning-how-the-director-chooses)).
# **Options:** ε-greedy (separate table per room type), LinUCB, Thompson (LinTS), each with a HumorGen,
# uniform or wrong (shuffled) prior.
# 
# **How to read:** a **career** = 300 shows in random rooms; 10 careers per policy. *First 10 shows* =
# cold start; *last 50* = experienced; *laughs lost vs oracle* = total over the career, **lower is
# better**. In the figure, a line that **bends flat** is learning; a straight line never learns.
# Overlapping ± ranges = tie.

# In[7]:


pol = run_or_load("policies", evaluate_policies.run, rerun=RERUN, shows=300, seeds=10)
show(evaluate_policies.table(pol))
fig = evaluate_policies.plot(pol)


# In[8]:


lin, eps = pol["LinUCB | uniform"], pol["ε-greedy | uniform"]
neu, orc = pol["fixed neurotic"], pol["oracle"]
check(f"Experienced LinUCB gets ≥ 95% of the oracle ({lin['warm'][0] / orc['warm'][0]:.0%})",
      lin["warm"][0] / orc["warm"][0] >= 0.95)
check(f"LinUCB beats always-Neurotic by ≥ 15% once experienced (+{lin['warm'][0] / neu['warm'][0] - 1:.0%})",
      lin["warm"][0] / neu["warm"][0] >= 1.15)
check(f"Pooling matters: ε-greedy loses ≥ 2× more laughs ({eps['lost_total'][0] / lin['lost_total'][0]:.1f}×)",
      eps["lost_total"][0] >= 2 * lin["lost_total"][0])
lo = lambda r: r["lost_total"][0] - r["lost_total"][1]
hi = lambda r: r["lost_total"][0] + r["lost_total"][1]
lins = [pol[f"LinUCB | {p}"] for p in ("humorgen", "uniform", "wrong")]
check("The prior makes no significant difference for LinUCB (CIs overlap)", max(map(lo, lins)) <= min(map(hi, lins)))
check("Uniform has the lowest loss among LinUCB priors", min(lins, key=lambda r: r["lost_total"][0]) is lins[1])


# ## Exp 5 Mid-set preference shift (2026-10-28)
# 
# **Decision:** keep the director's features; reacting within a show needs show state (MDP), not a
# per-bit reaction feature.
# **Where it fits:** can the director notice a room *turning*? ([simulated audience](../docs/wiki/simulated-audience.md))
# **Options:** LinUCB / LinTS with and without the **reaction feature** (same mode × last reward).
# 
# **How to read:** in a random half of shows the room's taste shifts after bit `shift_at`.
# "% of oracle" in **shifted** vs **steady** shows = what the turn costs. **Bits to recover**:
# lower is better; compare with **random**: near it means recovering by luck, not by noticing.

# In[9]:


sh = run_or_load("shift", evaluate_shift.run, rerun=RERUN, shows=400, seeds=10, shift_at=4)
show(evaluate_shift.table(sh))


# In[10]:


pct = lambda name, kind: sh[name][kind]["post"][0] / sh["oracle"][kind]["post"][0]
check(f"Steady shows: LinUCB ≥ 93% of oracle ({pct('LinUCB', 'steady'):.0%})", pct("LinUCB", "steady") >= 0.93)
check(f"Shifted shows: LinUCB drops below 85% ({pct('LinUCB', 'shifted'):.0%})", pct("LinUCB", "shifted") < 0.85)
rec = lambda n: sh[n]["shifted"]["recover"]
check("Learners recover no faster than random (within 0.3 bits)",
      all(rec(n)[0] > rec("random")[0] - 0.3 for n in ("LinUCB", "LinTS")))
gain = sh["LinUCB + react"]["shifted"]["post"][0] - sh["LinUCB"]["shifted"]["post"][0]
noise = sh["LinUCB + react"]["shifted"]["post"][1] + sh["LinUCB"]["shifted"]["post"][1]
check(f"Reaction feature gain ({gain:+.1f} laughs) is within noise (±{noise:.1f})", abs(gain) <= noise)


# ## Exp 6 Does a longer set let the director read a turn? (2026-10-28)
# 
# **Decision:** keep 8-bit sets (a 10-minute show); longer sets don't fix within-show adaptation.
# **Where it fits:** follow-up to Exp 5. After the shift the director had only 4 bits left, so maybe it
# just ran out of time? ([simulated audience](../docs/wiki/simulated-audience.md))
# **Question:** with more bits after the shift, does the director climb back to its steady-show level?
# **Options:** sets of 8, 16, 32, 52 bits (52 = a full hour special; Bargatze has 52), shift at the
# midpoint. **Every career gets the same 3,200 bits of experience**, so longer sets mean fewer
# shows, not more data. Evaluated on the last quarter of each career.
# 
# **How to read:** *choice quality* = expected reward of the chosen mode ÷ the best mode's (1.0 = best
# pick). It's noise-free, so it shows *decisions*, not luck. In the figure, solid = shows where the room
# turned, dotted = steady shows. If the director reads the room, the solid line climbs back to the
# dotted one after the shift. **Bits to recover** is shown for comparison but is misleading here:
# random "recovers" fast by luck, and a deterministic policy that sticks to its favorite looks worse.

# In[11]:


lengths = run_or_load("shift_set_length", evaluate_shift.run_lengths, rerun=RERUN,
                      lengths=[8, 16, 32, 52], total_bits=3200, seeds=10)
show(evaluate_shift.table_lengths(lengths))
fig = evaluate_shift.plot_lengths(lengths)


# In[12]:


import numpy as np
def late(n, name, key="curve"):
    c, k = lengths[str(n)][key][name], int(n)
    return float(np.mean(c[k - max(1, k // 4):]))
def just_after(n, name):
    r = lengths[str(n)]
    return float(np.mean(r["curve"][name][r["shift_at"]:r["shift_at"] + 2]))
pct = lambda n, name: lengths[str(n)]["summary"][name]["shifted"]["post"][0] / lengths[str(n)]["summary"]["oracle"]["shifted"]["post"][0]

check("Even 26 bits after the shift, LinUCB's choices stay ≥ 0.1 below its steady-show level (all lengths)",
      all(late(n, "LinUCB", "steady_curve") - late(n, "LinUCB") >= 0.1 for n in (8, 16, 32, 52)))
check(f"Longer sets barely help: after-shift laughs go {pct(8, 'LinUCB'):.0%} → {pct(52, 'LinUCB'):.0%} of oracle (< 5 points)",
      pct(52, "LinUCB") - pct(8, "LinUCB") < 0.05)
check(f"It does drift back slowly: 52-bit LinUCB climbs {just_after(52, 'LinUCB'):.2f} → {late(52, 'LinUCB'):.2f} (≥ 0.05)",
      late(52, "LinUCB") - just_after(52, "LinUCB") >= 0.05)
rec = lambda n, name: lengths[str(n)]["summary"][name]["shifted"]["recover"][0]
check("'Bits to recover' misleads: deterministic LinUCB looks worse than random at 32+ bits",
      all(rec(n, "LinUCB") > rec(n, "random") for n in (32, 52)))


# ## Exp 7 Show state, step A: does remembering tonight help? (2026-10-28)
# 
# **Plan (user):** add show state one step at a time and measure each: **A** tonight's record as
# features → **B** two timescales → **C** MDP. This is step A.
# **Where it fits:** Exp 5–6 showed the director can't read a room turning, because everything it knows
# is a career-long average. ([glossary: show state](../docs/wiki/glossary.md#learning-how-the-director-chooses))
# **Change:** for each mode, two more inputs: *tried tonight* (0/1) and *mean reward tonight*. The
# career model learns how much to trust them. Nothing else changes.
# 
# **Pass criteria, written before running** (so the result can't bend them):
# 1. **Shifted shows:** mean choice quality after the shift improves by **≥ 0.05**.
# 2. **Steady shows:** quality over the same bits drops by **no more than 0.01** (no chasing noise).
# 
# **How to read:** dashed = without show state, solid = with; orange = LinUCB, green = LinTS. A solid
# line above its dashed one after the shift = remembering tonight helps. The dotted grey line is
# steady-show quality, the level a full recovery would reach. Same harness as Exp 6 (3,200 bits per
# career, last quarter), 8- and 16-bit sets.

# In[13]:


from sim.evaluate_shift import SHOW_STATE_POLICIES, show_state_deltas
step_a = run_or_load("show_state_a", evaluate_shift.run_lengths, rerun=RERUN, lengths=[8, 16],
                     total_bits=3200, seeds=10, names=list(SHOW_STATE_POLICIES))
show(evaluate_shift.table_show_state(step_a))
fig = evaluate_shift.plot_show_state(step_a)


# In[14]:


deltas = show_state_deltas(step_a)
ok = lambda d: d["shifted_gain"] >= 0.05 and d["steady_change"] >= -0.01
show("| Set | Algorithm | Criterion 1: shifted gain ≥ 0.05 | Criterion 2: steady change ≥ −0.01 | Passes both? |\n|---|---|---|---|---|\n"
     + "\n".join(f"| {d['bits']} | {d['algo']} | {d['shifted_gain']:+.3f} | {d['steady_change']:+.3f} | "
                  f"{'yes' if ok(d) else 'no'} |" for d in deltas))
# The decision is "don't adopt A"; these checks hold while that stays true.
check("Decision holds: step A passes both pre-set criteria in no cell → not adopted", not any(map(ok, deltas)))
check("A helps a little after a turn (gain > 0 in every cell): direction right, size too small",
      all(d["shifted_gain"] > 0 for d in deltas))
check("A costs something in steady rooms (change < 0 in every cell): the noise-chasing trade-off",
      all(d["steady_change"] < 0 for d in deltas))


# ## Adding a new decision
# 
# 1. Put the experiment in a module as `run(**params) -> dict` and `table(results) -> str`
#    (plus `plot` if it has a figure), and call it via `run_or_load(name, run, **params)`.
# 2. Add a section here: **Decision · Where it fits · Question/options · How to read**, a cell that
#    runs it, and **checks** for every claim the decision makes.
#    Checks assert the **decision**, including rejections ("option X misses the pre-set criteria → not
#    adopted"), so all ✅ always means "every decision still holds". Show per-option pass/fail as a table.
# 3. Add the title fragment to `EVIDENCE` in the index cell, and the entry to `docs/decisions.md`.
# 4. `python scripts/run_notebook.py`, then commit the notebook with its outputs.
