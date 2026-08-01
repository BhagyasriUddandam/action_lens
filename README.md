# ActionLens

**Which video-AI approach should you use — a specialized action model, or a VLM?**

ActionLens answers that with real numbers. It classifies human actions in short video
clips two ways — (a) a pretrained pose/action-recognition baseline, (b) a vision-language
model — and benchmarks them head to head on accuracy, failure cases, latency, and cost.

Built by Bhagya Sri Uddandam.

> **Status: Checkpoint A, scaffold complete.** No pipeline code yet. Sections marked
> _TODO_ get filled in as each stage lands.

---

## The question

Specialized models are fast and cheap but narrow. VLMs are flexible and general but
slower and costlier. For "physical AI" work — fall detection, patient monitoring,
factory-floor activity — teams have to pick one. This repo measures the tradeoff on a
small, honest benchmark instead of guessing.

**Headline finding:** _TODO — one plain sentence, filled in at Checkpoint B._

---

## Results

_TODO — Checkpoint B._

| | Baseline (pose/action) | VLM |
|---|---|---|
| Accuracy | — | — |
| Latency / clip | — | — |
| Cost / 1k clips | — | — |

Full write-up: `FINDINGS.md` (Checkpoint B). Raw numbers: `results/metrics.csv`.

---

## Repo map

```
actionlens/
├── README.md
├── FINDINGS.md            # Checkpoint B — the written analysis
├── requirements.txt
├── run.slurm              # HPC job template (single GPU)
├── src/
│   ├── data_prep.py       # clips → sampled frames + labels.csv
│   ├── baseline_model.py  # pretrained pose/action model → predictions
│   ├── vlm_benchmark.py   # frames → VLM → labels, with latency + cost logging
│   └── evaluate.py        # both models → metrics, failure cases, plots
├── data/
│   ├── raw/               # source videos (gitignored)
│   ├── clips/             # trimmed clips (gitignored)
│   └── frames/            # sampled frames (gitignored)
├── results/
│   ├── metrics.csv
│   ├── results.json       # the dashboard reads this
│   └── plots/
├── logs/                  # SLURM stdout/stderr
├── dashboard/             # Checkpoint C — React + Vite, built LAST
└── tasks/
    ├── todo.md
    └── lessons.md
```

`src/`, `dashboard/`, and `FINDINGS.md` do not exist yet — they arrive with their checkpoints.

---

## Scope

Deliberately small, so it finishes:

- **Classes:** 4 — walking, sitting, standing, falling
- **Clips:** ~30–60 per class
- **Models:** pretrained only. Nothing is trained from scratch.
- **Dashboard:** static — reads `results/results.json`. No backend, no database, no auth.

---

## Setup — Northeastern Explorer HPC

Heavy work runs on HPC, not a laptop.

```bash
# 1. log in and get to the repo
ssh <username>@login.explorer.northeastern.edu
git clone <repo-url> actionlens && cd actionlens

# 2. create the conda env  (VERIFY the module name first: module avail anaconda)
module load anaconda3/2024.06
conda create -n actionlens python=3.11 -y
source activate actionlens

# 3. install torch against the cluster's CUDA, THEN everything else
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4. SLURM writes here — the job fails immediately if this is missing
mkdir -p logs
```

**Sanity-check the GPU before submitting real work:**

```bash
srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

_TODO: note where clip data lives (`/scratch/$USER/...`) once `data_prep.py` exists._

---

## Running the pipeline

`run.slurm` takes the stage name as its argument. Run them in order — each depends on
the previous one's output.

```bash
sbatch run.slurm data_prep        # Checkpoint A
sbatch run.slurm baseline_model   # Checkpoint A
sbatch run.slurm vlm_benchmark    # Checkpoint B
sbatch run.slurm evaluate         # Checkpoint B

squeue -u $USER                   # watch the queue
tail -f logs/actionlens-<jobid>.out
```

Before your first submit, open `run.slurm` and check the three lines marked `VERIFY`
(partition, GPU constraint, anaconda module version) against what the cluster actually
reports — these change between cluster upgrades.

### Local (Mac)

Only for editing code and, later, running the dashboard. If you do run a model locally,
PyTorch picks the MPS backend on Apple Silicon — expect it to be slow. Default to HPC.

---

## Dashboard

Checkpoint C. Not started, and intentionally so: it does not get built until
`results/results.json` exists. Setup instructions land here then.

---

## Reproducing this

_TODO — once the pipeline runs end to end, this section must let a stranger go from
`git clone` to the numbers in the results table above with no extra guidance._
