# TRD — Technical Requirements Document
## Project: "ActionLens"

Companion to PRD.md. Defines the tech stack, architecture, and technical constraints.

---

## 1. Architecture Overview
```
[ Video Clips ]
      |
      v
[ data_prep.py ] -- sample clips, extract frames, build labels.csv
      |
      +---------------------------+
      v                           v
[ baseline_model.py ]      [ vlm_benchmark.py ]
  pretrained pose/          VLM classifies sampled
  action recognition        frames via API
      |                           |
      +------------+--------------+
                   v
            [ evaluate.py ] -- metrics, failure cases, latency, cost
                   |
                   v
            [ results/ ] metrics.csv + plots
                   |
                   v
            [ Web Dashboard ] -- reads results, displays visually
```

## 2. Tech Stack

### Backend / ML (the substance)
- **Language:** Python 3.11+
- **ML:** PyTorch, torchvision
- **Video/CV:** OpenCV, MediaPipe (pose extraction)
- **Baseline model:** a PRETRAINED action-recognition or pose-based classifier (no training from scratch)
- **VLM:** a vision-language model via API (frames -> action label)
- **Data/plots:** NumPy, pandas, matplotlib
- **Compute:** Northeastern Explorer HPC (SLURM, single GPU node)
- **Env:** conda or venv on HPC

### Frontend / Dashboard (the wow) — build ONLY after ML works
- **Framework:** React (latest) + Vite
- **Styling:** Tailwind CSS (Franco-standard utility approach)
- **Animation:** GSAP + Animate UI for smooth, premium feel
- **Principles:** SOLID applied to component design (small, single-responsibility components)
- **Package manager:** bun (not npm)
- **Node:** latest LTS (note: "Node 16" is EOL/insecure — use current LTS instead; see Lessons)
- **Data source:** reads the pipeline's results (metrics.csv / a results.json) — static, no live backend needed

## 3. Repo Structure
```
actionlens/
  README.md
  FINDINGS.md
  requirements.txt          # python
  run.slurm                 # HPC job script
  src/
    data_prep.py
    baseline_model.py
    vlm_benchmark.py
    evaluate.py
  results/
    metrics.csv
    results.json            # dashboard reads this
    plots/
  dashboard/                # built LAST
    (React + Vite + Tailwind + GSAP app)
  tasks/
    todo.md
    lessons.md
```

## 4. Technical Constraints
- Pretrained models only. No from-scratch training.
- Small data: 3–5 classes, ~30–60 clips each.
- Dashboard is static (reads a results file) — no server, no DB, no auth.
- Everything reproducible: README must let someone re-run from scratch.

## 5. Environment Notes (MacBook M4 Pro + HPC)
- Heavy ML/GPU work runs on **HPC**, not the Mac.
- The **Mac** is for: writing code, running the dashboard locally (bun + Vite), recording the demo.
- Apple Silicon (M4): if you ever run models locally, use PyTorch MPS backend — but default to HPC.

## 6. Key Decisions & Rationale
- **Static dashboard** (reads a file) instead of a live backend: far simpler, no budget spent on server code, still looks great.
- **Pretrained + tiny data:** guarantees the project finishes; C10's #1 deliverable is a working reproducible pipeline, not scale.
- **ML before frontend:** substance first so there's always a credible thing to show.
