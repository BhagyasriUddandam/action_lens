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

- **2026-07-31 — Metered API work needs a resume cache before the first real call.**
  vlm_benchmark.py spends real money per clip, so a crash 140 clips into a 160-clip
  run would re-pay for all of it. Built an append-only JSONL flushed after every
  completed clip, and verified the resume path before running anything: re-run made
  0 paid calls, partial cache re-ran exactly the missing clips, and a truncated final
  line survived. **Rule:** when a loop costs money or time per iteration, make it
  resumable and *test the resume* before the first paid run — and isolate per-item
  errors so one failure never aborts the batch.

- **2026-07-31 — A stub for offline testing should validate, not just return.**
  Without an API key, the dry-run stub could have just returned canned responses. Making
  it assert on request shape instead (base64 decodes, JPEG magic bytes, block ordering,
  required fields) turned it into a real test — it would have caught a malformed image
  block on a laptop instead of as a paid 400 mid-run. **Rule:** a test double for a
  paid/remote service should enforce that service's contract, so offline runs catch
  the errors the real call would.

- **2026-07-31 — Choose models and hyperparameters on calib, never on eval.** Two
  selections were needed for the baseline: which backbone, and which classifier head.
  The backbone was picked on *published* Kinetics-400 accuracy (an a-priori criterion
  that never touches our data) and the head by leave-one-out CV on the 32 calib clips.
  Comparing either on the eval split would have made the headline accuracy optimistic
  in a way no reader could detect. **Rule:** every choice made after seeing eval
  numbers silently becomes part of the fit. Decide on published metrics, or on a
  held-out calib split, and report eval exactly once. Sanity check that it worked:
  calib accuracy was 100% while eval was 60.9% — if the split were leaking, they'd
  converge.

- **2026-07-31 — Looked at the data, not just the counts.** data_prep.py reported a
  clean 160 clips / 40 per class. Rendering contact sheets and actually LOOKING showed
  a `walk` clip whose first 2 of 16 frames were a building exterior with no person —
  the clip spanned a shot cut. Measured after: 18-38% of HMDB51 clips per class.
  **Rule:** for any data pipeline, render a sample and inspect it before declaring
  success. Row counts and file counts confirm the code ran, not that the data is right.

- **2026-07-31 — Picked a detector without validating it on the failing case.** First
  cut detector used HSV histogram correlation < 0.5; it scored the known-bad Oceans12
  clip at 0.86 and passed it straight through, so the filter looked like it worked
  (it rejected 4 other clips) while missing the exact clip that motivated it. Mean
  absolute pixel difference separated the same cut 5-10x cleanly. **Rule:** when
  adding a detector/filter, first assemble the specific case that motivated it and
  confirm the metric actually separates it. A filter that fires on *something* reads
  as working. Also prefer a threshold adaptive to each item's own baseline over a
  global constant, since per-clip motion varies widely.

- **2026-07-31 — Cache path and fresh path disagreed on a written column.** `width`/
  `height` in labels.csv held post-resize dims after a fresh extract but source dims
  after a cached re-run — the file's meaning depended on invisible cache state. Caught
  only by diffing a fresh run against a re-run. **Rule:** when a function has a
  skip-if-cached branch, diff the full output of fresh-vs-cached runs, and make both
  branches converge on ONE code path for anything they both report.

- **2026-07-31 — "Server is blocking us" was actually a dead link.** The HMDB51 Brown
  URL returned HTML, which looked like user-agent blocking. It wasn't: Brown migrated
  the lab site off WordPress to GitHub Pages, so the host 301s to the new homepage and
  the old path 404s. UA/referer spoofing was never going to work. **Rule:** before
  writing UA-spoofing or scraping workarounds, follow the redirect chain
  (`curl -sIL`) and check whether the resource still exists. Distinguish
  301→homepage (dead link) from 403/captcha (actual bot block) — only the latter
  justifies a workaround. Also: prefer a mirror that stores individual files over one
  that stores archives — HF `CVML-TueAI/HMDB51` cut a 2.1 GB rar-of-rars download to a
  292 MB per-class fetch with no extraction tooling at all.

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
