# lessons.md — Mistakes to Never Repeat

Claude reads this at the start of every session and appends to it after any user correction.
Format: **Date — Mistake — Rule to prevent it.**

---

## Seeded lessons (known before we start)

- **Wrong-project stack.** Mistake: mixing web-app tooling (React/Tailwind/GSAP/bun) into the ML pipeline phase. Rule: the pipeline is Python/PyTorch ONLY. Frontend tooling appears ONLY in the dashboard phase, after the pipeline produces results.

- **Node 16 is insecure/EOL.** The user asked for "Node 16" but it reached end-of-life in 2023 and receives no security updates. Rule: use the current Node LTS instead, and briefly tell the user why when it comes up. Never scaffold on Node 16.

- **From-scratch training blows the budget.** Rule: pretrained models only. If a step implies training from scratch, STOP and propose a pretrained alternative.

- **Scope creep kills small projects.** Rule: hold to 3–5 classes, ~30–60 clips each. If asked to expand, flag the budget/time cost first.

- **Frontend-before-substance risk.** Rule: never begin the dashboard until `results/results.json` exists. Substance (ML) must be sendable on its own first.

- **Big-bang generation wastes budget.** Rule: build one phase at a time; verify each before moving on.

---

## Session lessons (append below as we learn)

- **2026-07-31 — Assumed dataset contents without checking.** The plan called for
  UCF101 to supply walk/sit/stand; UCF101 has none of those classes. Nearly built a
  loader for data that doesn't exist. **Rule:** before designing around any dataset,
  verify the actual class list / file layout / license against the official source
  page. Applies to modality too — "RGB dataset" may mean depth-primary (URFD) or
  still images rather than clips (the Roboflow URFD mirrors).

- **2026-07-31 — Paraphrased WebFetch summaries hid the wrong filename.** Planned
  URFD fall-windowing against `fall-XX-data.csv` per sequence, described (from a
  fetched summary) as holding −1/0/1 pose labels. Actually downloading it showed
  3 unlabeled numeric columns — it's a camera/accelerometer sync log. The real
  per-frame labels live in one aggregated file, `urfall-cam0-falls.csv`, covering
  all 30 sequences. **Rule:** for any file whose *content/schema* the plan depends
  on (not just its existence), download and inspect a real sample — a fetched page
  summary can name the wrong file with full confidence. Confirmed here by literally
  diffing frame-label transitions (83→0, 113→1) against the PNG count.

- **2026-07-31 — Single-source classes create a hidden confound.** Sourcing `falling`
  only from URFD would make source a perfect proxy for label, so a model could score
  well by recognizing the room. **Rule:** when combining datasets, check whether any
  class maps 1:1 to a source. If it does, mix sources for that class or report the
  confound explicitly — never quote a headline accuracy that the confound inflates.
