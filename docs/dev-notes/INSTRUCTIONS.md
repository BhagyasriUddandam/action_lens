# INSTRUCTIONS.md — Claude Working Agreement for "ActionLens"

You are helping Bhagya Sri Uddandam build ActionLens (see PRD.md + TRD.md). Follow these rules on EVERY task. Read `tasks/lessons.md` at the start of each session.

---

## PROJECT-SPECIFIC GUARDRAILS (read first)
- This is a **two-part project**: (1) a Python/PyTorch ML video-benchmark pipeline, (2) a React/Tailwind/GSAP dashboard built ONLY after the pipeline works.
- **Sequence is mandatory:** ML pipeline first (Checkpoints A & B), dashboard last (Checkpoint C). Never start the dashboard before the pipeline produces a results file.
- **Pretrained models only.** Never train from scratch. If a task implies from-scratch training, STOP and flag it.
- **Small scope:** 3–5 action classes, ~30–60 clips each. If scope creeps, STOP and re-plan.
- Heavy compute runs on **Northeastern HPC (SLURM)**, not the Mac. Provide SLURM scripts.
- The user can follow technical steps but is newer to web tooling — explain web/React steps a bit more, keep ML steps concise.

## Workflow Orchestration

### 1. Plan First (default)
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- Write the plan to `tasks/todo.md` with checkable items BEFORE coding.
- Check in with the user before starting implementation.
- If something goes sideways, STOP and re-plan — don't keep pushing.

### 2. Subagent Strategy
- Use subagents for research, exploration, and parallel analysis to keep the main context clean.
- One task per subagent for focused execution.

### 3. Self-Improvement Loop
- After ANY correction from the user: append the pattern to `tasks/lessons.md`.
- Write a rule for yourself that prevents that mistake recurring.
- Review `tasks/lessons.md` at the start of every session.

### 4. Verify Before Done
- Never mark a task complete without proving it works (run it, show output/logs).
- Ask: "Would a staff engineer approve this?"
- For the pipeline: show it runs end-to-end. For the dashboard: show it renders.

### 5. Demand Elegance (Balanced)
- For non-trivial changes, pause: "Is there a simpler, more elegant way?"
- Skip this for simple obvious fixes — don't over-engineer.

### 6. Autonomous Bug Fixing
- Given an error/log/failing test: just fix it. Point at the cause, resolve it.
- Don't ask for hand-holding on routine bugs.

## Task Management
1. Plan first → write to `tasks/todo.md` (checkable items).
2. Verify plan → check in before implementing.
3. Track progress → mark items complete as you go.
4. Explain changes → high-level summary at each step.
5. Document results → add a review section to `tasks/todo.md`.
6. Capture lessons → update `tasks/lessons.md` after corrections.

## Core Principles
- **Simplicity first:** every change as simple as possible; minimal code impact.
- **No laziness:** find root causes; no temporary hacks; senior-developer standard.
- **Minimal impact:** only touch what's necessary; avoid introducing bugs.

## Frontend Rules (dashboard phase only)
- React (latest) + Vite, Tailwind CSS, GSAP + Animate UI.
- Apply SOLID: small single-responsibility components, no giant files.
- Package manager: **bun** (not npm).
- Node: **use current LTS**, NOT Node 16 (it is end-of-life and insecure — see lessons.md).
- Dashboard is static: it reads `results/results.json`. No backend/DB/auth.

## Budget Awareness (user has limited Claude Code hours)
- Be token-efficient: do ONE phase at a time; don't generate everything at once.
- Don't re-explain the whole plan each turn — reference PRD/TRD.
- Prefer concise, working code over long prose.
