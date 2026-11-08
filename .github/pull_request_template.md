## Where this fits
Task ID from `docs/plans/live-loop.md`, which part of the agent it touches, and the question it answers.

## What changed
One task only.

## Number that decided this
The number and pass criterion, as fixed before running. For a PR with no experiment, write "No experiment" and why.

## Results
Table with paired 95% CIs, and how to read it. Say which `audience.yaml` values the result depends on.

## Pre-merge review
Findings from the pre-review and how each was fixed. Which number was re-derived, and what it came to.

## Checklist
- [ ] One task, reviewable in 30 minutes or less, not stacked
- [ ] Only touches the files its task names, or says why not
- [ ] Root tests pass: `.venv/Scripts/python -m pytest tests -q`
- [ ] Backend tests pass in `backend/.venv`
- [ ] Tests check what the code guarantees
- [ ] Frozen contract unchanged, or a new decisions.md entry
- [ ] Experiment is a `run(**params)` through `run_or_load`, or N/A
- [ ] Section in the right `notebooks/decisions-*.ipynb` (data, director, live) with ✅/❌ checks, re-run, or N/A
- [ ] `COVERAGE` entry in `notebooks/decision-plot.ipynb`, re-run, or N/A
- [ ] Results in `docs/metrics.md`, or N/A
- [ ] Decision in `docs/decisions.md`, or "judgment call, not measured"
- [ ] Bugs that cost time in `docs/dev-log.md`
- [ ] New terms in `docs/wiki/glossary.md`
- [ ] `docs/status.md` and folder READMEs current
- [ ] No rJokes text, transcript text, session data or secrets committed
- [ ] Invariants in `CLAUDE.md` hold
