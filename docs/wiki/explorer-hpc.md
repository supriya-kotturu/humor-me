# Explorer HPC

**Where it fits:** offline GPU work that the laptop (RTX 4060) and Gemini's free tier can't
do in time. Nothing in the live loop or the sim depends on it.
**Used in:** `explorer/` (labeling job, batch script, env), later the joke scorer.

## Concepts

| Term | Meaning here |
|---|---|
| **Slurm** | The scheduler: you ask for resources, it queues and runs your job |
| **Partition** | A pool of nodes with limits. Ours: `courses` (CPU) and `courses-gpu` |
| **`sbatch`** | Submit a script to run unattended; survives closing the browser |
| **`srun` / OOD JupyterLab** | Interactive sessions; good for smoke tests, wasteful for long runs |
| **OOD** | Open OnDemand, the web portal (ood.explorer.northeastern.edu) |
| **vLLM** | Batched LLM inference server/library; why 30k labels take minutes, not days |

## Limits that shaped our design (CS5180 course access)

- `courses-gpu`: 8 h max, **1 running job**, **12 GB RAM**; idle ~1 h → killed.
  → long runs use `sbatch`; the script loads only the sample, not 400k jokes.
- Student dir `/courses/CS5180.<term>/students/<user>` is **deleted after grades**.
  → copy results back after every run; outputs hold ids and labels, never joke text.

## Nodes and storage: designing jobs that survive

- **No fixed node.** Slurm places each job on any free node in the partition, so a re-run may land
  elsewhere. State must live on a **shared filesystem**, never on the node.
- **Shared filesystems** (mounted everywhere): `/courses/...` (ours; gone after grades), `/home`
  (75 GB, permanent), `/scratch` (purged monthly), `/projects` (permanent, needs a PI allocation).
  Node-local `/tmp` dies with the job. There is **no S3-style object storage**.
- **Checkpoint by output, not by node:** our labeling job appends results in flushed chunks and,
  on start, skips ids already in its output. Re-submitting the same command resumes; a fixed seed
  plus a larger `--n` extends the same sample. This is the general pattern for any long job here.

## Further reading

- [NURC course guide](https://rc-docs.northeastern.edu/en/explorer-main/classroom/class_guide.html):
  partitions, storage, OOD.
- [NURC scratch purge policy](https://rc-docs.northeastern.edu/en/explorer-main/best-practices/scratchpurge.html):
  why nothing we keep lives in `/scratch`.
- [Slurm quick start](https://slurm.schedmd.com/quickstart.html): `sbatch`, `squeue`,
  `scancel`.
- [vLLM docs](https://docs.vllm.ai/): offline batch inference (`LLM.chat`).
