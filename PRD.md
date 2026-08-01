# PRD — Product Requirements Document
## Project: "ActionLens" — Video Action Recognition Benchmark + Dashboard

**Author:** Bhagya Sri Uddandam
**Purpose:** A portfolio project to send to Shahid Azim (CEO, C10 Labs) demonstrating I can do the C10 ML & Computer Vision co-op — running video models, applying VLMs, benchmarking them, and presenting results clearly.
**Audience for the deliverable:** A CEO (business + technical). Must be immediately understandable AND technically credible.

---

## 1. Problem Statement
C10 Labs builds "physical AI" — AI applied to the real, physical world (their examples: a copilot for nurses, smarter manufacturing). A recurring need is understanding **human activity from video** (e.g., detecting a fall, recognizing an action). Two approaches exist:
- **Specialized models** (pose/action-recognition) — fast, cheap, but narrow.
- **Vision-Language Models (VLMs)** — flexible, general, but slower and costlier.

Teams need to know **which approach to use when.** This project answers that with real numbers.

## 2. What We're Building
A two-part deliverable:
1. **ML Pipeline (the substance):** Classifies human actions in short video clips using (a) a pretrained pose/action baseline and (b) a VLM. Benchmarks both on accuracy, failure cases, latency, and cost.
2. **Web Dashboard (the wow):** A clean, animated single-page site that shows the video clips, both models' predictions side-by-side, and the benchmark charts — so a non-technical CEO instantly *sees* the value.

## 3. Goals & Success Criteria
- [ ] Pipeline runs end-to-end and reproducibly on Northeastern HPC (SLURM/GPU).
- [ ] Benchmarks baseline vs. VLM: accuracy, per-class breakdown, failure cases, latency, cost.
- [ ] Dashboard clearly presents results; loads fast; looks polished.
- [ ] Clean GitHub repo + README + FINDINGS write-up + a 2–3 min demo video.
- [ ] The core insight is stated plainly: "Use X when speed/cost matters, use VLM when flexibility matters."

## 4. Non-Goals (explicitly out of scope — protects budget/time)
- NOT training a model from scratch (use pretrained only).
- NOT a large dataset (3–5 classes, ~30–60 clips each).
- NOT a production app with auth/users/database.
- NOT real-time streaming — batch processing is fine.

## 5. Target Users
- Primary: C10 Labs CEO & team evaluating me as a candidate.
- Framed as: how a C10 portfolio startup would evaluate video-AI approaches.

## 6. Scope Classes (example)
Walking, Sitting, Standing, Falling (fall detection = strong "physical AI for health" story, matches C10's BioHub).

## 7. Deliverables
1. GitHub repo (code + README + FINDINGS.md)
2. Results: metrics.csv + plots
3. Web dashboard (deployed or screen-recorded)
4. 2–3 min demo video for the CEO email

## 8. Sequencing (always have something sendable)
- **Checkpoint A:** Pipeline + baseline works → you already have a real project.
- **Checkpoint B:** VLM benchmark added → now it's the full C10 workflow.
- **Checkpoint C:** Dashboard on top → now it's CEO-ready wow.
Each checkpoint is a complete, mailable state.
