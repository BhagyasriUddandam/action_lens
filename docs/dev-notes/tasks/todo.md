# todo.md — ActionLens Build Plan

Claude updates this as we go. Checkpoints = always-sendable states.

## CHECKPOINT A — Pipeline + Baseline (sendable as a real project)
- [x] Scaffold repo (folders, requirements.txt, README skeleton, run.slurm)
  - [x] Folder structure per TRD §3 (+ `data/`, `logs/`)
  - [x] requirements.txt (Checkpoint A deps + B section pre-listed)
  - [x] README skeleton with HPC setup + run order
  - [x] run.slurm — single-GPU Explorer template, stage-parameterized
  - [x] .gitignore
  - [ ] **User action:** verify the 3 `VERIFY` lines in run.slurm on Explorer
        (`sinfo -s`, `sinfo -o "%P %G %f"`, `module avail anaconda`)
- [ ] data_prep.py: get 3–5 activity classes, sample 30–60 clips each, extract frames, build labels.csv
  - **Datasets (decided 2026-07-31):** UCF101 REJECTED — it has no walk/sit/stand class
    (closest is `WalkingWithDog`). Replaced by **HMDB51**, which has `walk`, `sit`,
    `stand` AND `fall_floor` natively (CC BY 4.0, ~2GB, .avi).
    Roboflow URFD versions REJECTED — object-detection *still images*, no temporal
    sequences, unusable for action recognition.
  - **Class plan:** 4 classes × 40 clips = 160.
    walk/sit/stand ← HMDB51 (40 each); falling ← URFD cam0 (25) + HMDB51 fall_floor (15).
  - **Why falling is mixed-source:** if `falling` came only from URFD, source would
    correlate perfectly with label and a model could win by recognizing the room, not
    the fall. Mixing also yields a real finding: staged vs. real fall accuracy.
  - **Canonical format:** `data/frames/{class}/{source}_{clip_id}/frame_NN.jpg`
    (16 frames, uniform temporal sampling, shorter side 256) + `data/labels.csv`
    with columns `clip_id,label,source,source_path,n_frames,fps_orig,duration_s,split`.
    `source` column is what enables the per-source fall analysis.
  - **Design:** two adapters (`load_hmdb51_clip`, `load_urfd_sequence`) each return a
    plain frame list; sampling/resize/write/CSV are shared and source-agnostic.
    A third source later = one new adapter, nothing else.
  - **URFD windowing:** a fall sequence is mostly NOT falling (walk in → stand → fall →
    lie still). Must window around the transition using per-frame pose labels.
    Uniform sampling of the full sequence would label mostly-standing frames as
    `falling`.
  - **CORRECTED 2026-07-31 (was wrong in original plan):** the per-frame labels are
    NOT in each sequence's `fall-XX-data.csv` (that file is a cam/accelerometer sync
    log — frame_num, sync_value, accel_reading — not pose labels). The real labels
    are in ONE aggregated file for all 30 sequences: `urfall-cam0-falls.csv`
    (no header; columns `sequence,frame_num,label,<8 geometry features>`; label is
    −1 = not lying, 0 = falling/transition, 1 = lying). Verified against fall-01:
    160 label rows == 160 PNGs, frame N label lines up exactly with
    `fall-01-cam0-rgb-{N:03d}.png`, transition at frames 83 (label 0) → 113 (label 1).
    Also: the zip extracts one directory level deeper than the zip name
    (`fall-01-cam0-rgb/fall-01-cam0-rgb/*.png`) — adapter must glob recursively.
  - [x] `scripts/fetch_urfd.sh` — downloads all 30 fall sequences (RGB cam0 zips)
        + the single `urfall-cam0-falls.csv` label file. Idempotent (skips existing),
        resumable (`curl -C -`), continues past a single failed sequence, `--only`
        flag for testing, `--clean` to delete zips after extraction.
        Verified: ran end-to-end on sequences 01+02 (real download), confirmed
        idempotent re-run skips cleanly, confirmed extracted structure and label
        alignment by hand.
  - **Not using `torchvision.datasets.HMDB51`:** needs exact split-file layout, no
    auto-download, fold logic fights the 40-per-class cap. Glob class folders instead.
  - [x] `scripts/fetch_hmdb51.sh` — NO manual download needed after all.
        **Both official sources are DEAD (verified 2026-07-31):** the Brown URL
        `serre-lab.clps.brown.edu/wp-content/uploads/.../hmdb51_org.rar` 301s to the
        new GitHub-Pages homepage and 404s on the new domain (site migrated off
        WordPress) — this is NOT a user-agent block, browser UA/referer/-L all fail
        identically. The Google Drive mirror on the current lab page also 404s.
        **Source used:** HF `CVML-TueAI/HMDB51` — complete mirror (6766 clips, 51
        classes) storing INDIVIDUAL .avi under `hmdb51_org/{class}/`, ungated.
        So we fetch only our 4 classes (~292 MB / 980 clips) and need no unrar at
        all — note `unrar`/`7z` are NOT installed on this Mac anyway.
        Fallback `--method rar`: HF `Serrelab/hmdb51` (Brown's own account, the
        authoritative 2.1 GB rar-of-rars, confirmed live) — needs unrar, insurance only.
        Two download paths, both tested: `huggingface_hub.snapshot_download`
        (primary; copies out of cache so no symlinks leak downstream) and a plain
        curl loop (used when huggingface_hub is missing, e.g. bare env on Explorer).
        Verified: fall_floor 136/136 via hf path in 13s; curl fallback exercised with
        huggingface_hub forced unimportable, restored 3 deleted files, skipped 133;
        all 136 confirmed real AVI, 0 symlinks, 0 truncated.
        **Note for data_prep:** clip resolution VARIES (320x240, 416x240, 592x240
        seen) — the resize step is required, not optional.
  - [ ] **User action:** accept URFD CC BY-NC-SA 4.0 (non-commercial academic) —
        ACCEPTED 2026-07-31; must cite Kwolek & Kepski (2014) in README + FINDINGS
  - [x] Verify: labels.csv has 160 rows, 40/class, both sources present in `falling`
  - **BUILT 2026-07-31.** 160 clips / 2560 frames / 57 MB, full build 8.6s.
    walking 40, sitting 40, standing 40, falling 40 (urfd 25 + hmdb51 15).
    Splits stratified by (label, source): 128 eval / 32 calib.
  - **Two data-quality filters added after measuring the real data:**
    1. HMDB51 encodes quality in filenames (`_goo_`/`_med_`/`_bad_`). Excluding
       `bad` still leaves 92-496 clips per class vs the 40 needed. Free win.
    2. 18-38% of HMDB51 clips span a SHOT BOUNDARY (worst: fall_floor 38%), so
       sampled frames show two different scenes. 36 clips rejected in the final
       build; rejection draws a replacement so counts still hit target.
  - **Cut detector — histogram correlation FAILED, pixel difference works.**
    First attempt used HSV hue/sat histogram correlation < 0.5. It missed a real
    cut in `Oceans12_walk_..._12` (building exterior -> night street) because both
    scenes are warm-toned: correlation was 0.86. Same cut shows mean abs pixel
    diff 55.6 vs a 0.6-10.2 baseline. Final rule is adaptive: a pair is a cut if
    diff > 30 AND diff > 4x that clip's own median diff (motion baseline varies
    per clip). Measured over 200 clips: non-cut pairs sit at ratio 1.5-2.4,
    URFD's fixed camera peaks at 2.5, the Oceans12 cut is 11x.
  - **Verified:** all 2560 frames readable, every short side exactly 256, aspect
    preserved (widths 324-478). Determinism: same seed -> identical clip set AND
    order; seed 7 overlaps seed 42 on only 58/160. Idempotent re-run: 0 written,
    160 reused, 0.9s, byte-identical labels.csv. Visually confirmed one clip per
    class/source via contact sheets — including that the URFD window captures the
    full fall arc (upright -> falling -> on the ground), not the standing run-up.
  - **Bug found and fixed during verification:** `width`/`height` reported
    post-resize dims on a fresh extract but source dims on a cached re-run, so
    labels.csv silently changed meaning depending on cache state. Both paths now
    converge on `read_output_dims()` (reads the written frame back from disk),
    and source dims moved to separate `src_width`/`src_height` columns.
  - **Note for Checkpoint B prompting:** HMDB51's `sit` and `stand` are
    TRANSITIONS (sitting down / standing up), not static postures — same for
    `falling`. Prompt the VLM for the action, not the pose, or `standing` and
    `sitting` will be systematically confused.
- [x] baseline_model.py: run a PRETRAINED pose/action model, output per-clip predictions + accuracy
  - **Kinetics-400 has NONE of our classes** (verified against the K400 label list):
    no `standing`, no `falling`, no generic `walking`/`sitting` — only "walking the
    dog" and "situp". Using the pretrained classifier head with label mapping would
    score ~0 on 3 of 4 classes. Same trap as UCF101; checked before building this time.
  - **Design:** frozen pretrained backbone as a feature extractor + a 4-class head
    fitted ONLY on the 32 calib clips, scored on the 128 eval clips. Backbone weights
    are never updated — a linear probe, not from-scratch training. This also makes
    Checkpoint B fair: the VLM gets the same 32 calib clips for prompt tuning and the
    same 128 eval clips for scoring, so both get an equal adaptation budget.
  - **Backbone chosen a priori by published K400 top-1** (mvit_v2_s 80.8 > r2plus1d_18
    67.5 > r3d_18 63.2), NOT by our eval numbers — picking on eval would leak.
    Confirmed after the fact: mvit_v2_s 60.9% vs r3d_18 48.4%, so the criterion held.
  - **Head chosen by leave-one-out CV on calib** (centroid 34.4% vs logreg 37.5% ->
    logreg), again never on eval.
  - **RESULT: 60.9% overall on eval (78/128), chance = 25%.**
    falling 90.6% | walking 75.0% | sitting 40.6% | standing 37.5%
    Per-source falling (the confound check): urfd 100% (20), hmdb51 75% (12) —
    the model is NOT just memorising the URFD room, though real falls are easier.
    Latency batch=1 on MPS: mvit_v2_s 320ms vs r3d_18 70ms.
  - **Main failure mode, quantified:** sitting<->standing. Their class centroids sit at
    0.868 cosine similarity, far above every other pair (next 0.579) and higher than
    either class's own intra-class cohesion (0.406 / 0.356) — they occupy nearly the
    same region of feature space. Root cause is semantic: HMDB51 `sit`/`stand` are
    TRANSITIONS (sitting down / standing up) that are near time-reverses, and a K400
    backbone was never taught to encode direction of motion. r3d_18 confuses them
    just as badly, so this is the data + pretraining, not one model's quirk.
    This is the PRD's "specialised models are narrow" thesis with a number attached —
    do NOT paper over it; it is the setup for the VLM comparison.
  - **Verified:** deterministic across runs (identical predictions + metrics);
    metrics computed on eval rows only (recomputed 0.6094 == reported); calib
    accuracy is 100% (overfit as expected) and correctly excluded from reporting.
  - Outputs: `results/baseline_predictions.csv` (all 160 clips, split-tagged),
    `results/baseline_metrics.json`, optional `--features-out/--features-in` cache
    (gitignored) so re-runs skip the 52s embedding pass.
- [ ] Verify: pipeline runs end-to-end on HPC, prints accuracy
- [ ] Write README setup steps (yourself)

## CHECKPOINT B — VLM Benchmark (full C10 workflow)
- [~] vlm_benchmark.py: send sampled frames to a VLM, get action labels, log latency + approx cost
  - **WRITTEN AND VERIFIED OFFLINE 2026-07-31; NOT YET RUN — no API credentials on this
    machine** (`ANTHROPIC_API_KEY` unset, no `ant` CLI). Everything except the live
    call is tested; the run itself is blocked on the user.
  - **Model:** `claude-opus-5`, 8 of the 16 frames per clip (evenly spaced, endpoints
    always included — the first/last frames carry the direction-of-motion signal).
  - **Prompt is the experiment.** Asks for the ACTION, not the pose, and gives
    contrastive definitions: sitting = controlled descent onto a support,
    standing = controlled ascent to the feet, falling = UNcontrolled drop to the
    floor. States frames are chronological and tells the model to compare FIRST
    vs LAST frames, because sit-down and stand-up are near time-reverses and look
    identical mid-transition. This targets exactly the failure the baseline had
    (sit/stand centroids at 0.868 cosine similarity).
  - **Fair fight preserved:** prompt tuned only on the 32 calib clips
    (`--split calib`), scored once on the 128 eval clips — same split the baseline
    fits its head on, so both get an equal adaptation budget.
  - **Structured output** via `client.messages.parse()` + Pydantic
    (`action` enum / `confidence` / one-sentence `evidence`). The evidence field is
    cheap and gives real failure-case material for FINDINGS.
  - **Money safety:** append-only JSONL cache (`results/vlm_cache.jsonl`) flushed
    after every completed clip, so a crash / rate limit / Ctrl-C never re-pays.
    Cost computed from the API's own `usage` fields, not estimated. Prompt caching
    on the system block (0.1x reads). Per-clip errors and refusals are recorded and
    skipped rather than aborting the batch.
  - **Verified offline** with a stub client that also VALIDATES request shape
    (base64 decodes, JPEG magic bytes, block ordering, disabled-thinking/effort
    guard): full pipeline on 12 clips; resume made **0 paid calls** with identical
    predictions; partial cache re-ran exactly the 3 missing; a truncated final line
    (hard-kill sim) recovered; refusal and API-exception paths return cleanly with
    cost still recorded; cost math checked against hand-computed value.
  - **Estimated full-run cost ~$1.50–3** (stub projects ~$8.85/1k clips → ~$1.42 for
    160). Verify with `--limit 4` (a few cents) before the full run.
  - [ ] **User action:** provide credentials (`export ANTHROPIC_API_KEY=...` or
        `ant auth login`), then run `--limit 4`, then `--split calib`, then
        `--split eval`
  - [ ] Report per-class accuracy and whether the VLM separates sit from stand
- [x] evaluate.py: baseline vs VLM — accuracy, per-class, failure cases, latency/cost table + plots
  - **WRITTEN AND VERIFIED 2026-07-31, NOT YET RUN FOR REAL** — this machine has no
    `vlm_predictions.csv`/`vlm_metrics.json` (the VLM run happened elsewhere, likely
    Explorer). Verified structurally with a SYNTHETIC vlm_predictions.csv matching
    the real schema (deleted immediately after — never committed, never treated as
    a real result). Confirmed: baseline-side numbers in results.json are computed
    fresh from the REAL baseline_predictions.csv (matched the previously-verified
    75/40.6/37.5/90.6% per class, 60.9% overall) — only the synthetic VLM side was
    fake. `sit_stand.vlm_separates_sit_stand` boolean, `largest_gap_class`, and
    failure-case bucketing all computed correctly from data, none hardcoded.
  - Filters both prediction files to `split=="eval"` defensively regardless of how
    each was generated; takes the clip_id intersection and warns on mismatch rather
    than silently dropping or crashing.
  - Outputs: `results/metrics.csv` (wide comparison table), `results/failure_cases.csv`
    (full disagreement list for FINDINGS.md), `results/plots/*.png` (per-class bars,
    side-by-side confusion matrices, latency, cost — using the project's validated
    categorical palette: blue #2a78d6 baseline / orange #eb6834 VLM), `results/results.json`
    (dashboard's ONLY data source).
  - **Baseline cost honesty:** no fabricated "$0" baseline cost bar — baseline runs
    on owned GPU compute, so `cost_note` states that instead of a false per-clip
    dollar comparison against the VLM's metered API cost.
  - [ ] **User action:** get the real `results/vlm_predictions.csv` +
        `results/vlm_metrics.json` onto this machine (or run `evaluate.py` wherever
        vlm_benchmark.py actually ran), then run `python src/evaluate.py` for real
- [ ] Write FINDINGS.md (yourself): when to use which approach

## CHECKPOINT C — Dashboard (CEO wow)
- [x] Scaffold React+Vite app with bun, Tailwind, GSAP (current Node LTS)
  - React 19.2 + Vite 8.2 (via `bun create vite --template react`), Tailwind v4
    (the current `@tailwindcss/vite` plugin approach, no tailwind.config.js needed),
    GSAP 3.15, bun 1.3.10. Node v24 confirmed current LTS.
  - **Deviated from a literal reading of INSTRUCTIONS.md's "GSAP + Animate UI":**
    used GSAP alone. Animate UI is a shadcn-style copy-paste library built on
    Motion (Framer Motion), not GSAP — pulling both in means two animation
    engines doing overlapping jobs, which fails "simplicity first" for a single
    scoped page. GSAP alone (scroll-triggered reveals, count-up numbers, staggered
    bar-fill) delivers the "smooth, premium feel" ask. Flagged for the user;
    can add real Animate UI components later if they specifically want that
    library's look.
- [x] Components (SOLID, one job each): Hero, InsightBanner (the sit/stand
      headline), SectionHeading, AccuracyChart, ConfusionMatrixGrid/Pair,
      LatencyCostPanel, FailureCard/Gallery, StatTile, Footer
  - Charts are custom-built (div-based), not a charting library — kept full
    control over the project's validated palette and dataviz-skill mark specs
    (thin bars, rounded ends, legend for 2+ series, sequential ramp for the
    confusion matrix) rather than fighting a library's defaults.
  - `dashboard/scripts/sync-data.mjs` (bun run sync, wired into dev/build)
    copies `results/results.json` into `public/` and exports first+last-frame
    thumbnails for the curated failure cases into `public/clips/` — frame
    images are gitignored (2560 files), so this is a deliberate small export,
    not full clip playback. Documents the pipeline/dashboard boundary as one
    command.
- [x] Load results.json; animate charts with GSAP
  - Count-up numbers (StatTile), scroll-triggered reveals (useInView +
    IntersectionObserver), staggered bar-fill (AccuracyChart), hero entrance
    timeline.
- [x] Verify it renders and looks polished
  - **Real browser verification**, not just HTTP checks: installed Playwright
    + Chromium (temporarily — removed after), drove the actual dev server,
    screenshotted every section, confirmed zero console errors.
  - **Two real bugs found and fixed during this pass:**
    1. First screenshot attempt (800ms wait) showed the Hero's two headline
       numbers as blank — traced to the GSAP entrance timeline taking ~1.4s
       total; the verification script's wait was too short, not a product bug.
       Confirmed via DOM inspection (`$$eval`) that the correct values were
       present in the DOM the whole time.
    2. A `<>` fragment shorthand with a `key` prop in ConfusionMatrixGrid
       (React silently rejects keys on shorthand fragments) — fixed with an
       explicit `Fragment` import before it ever hit the browser.
  - **Data-quality issue found in the VLM's own output, not the dashboard:**
    2 of 128 eval predictions' `evidence` text (free-text field only —
    `pred`/`confidence` unaffected, verified) had the model running on past
    the field boundary with a glued-on fragment (one with stray `{}`/`[]`,
    one a mid-sentence non-space period). Did NOT rewrite or fabricate
    cleaner text for real model output. Instead improved evaluate.py's
    curation (already picking 3 of 45 examples) to prefer clean candidates
    when enough exist — `has_clean_evidence()` in evaluate.py. Regenerated
    results.json; all reported numbers unchanged, since neither anomaly
    touched a scored field.
  - Confirmed final render top-to-bottom: Hero (60.9%/92.2%, real counted-up
    numbers), InsightBanner (real +53pt/+50pt sit/stand story), AccuracyChart
    (all 4 classes, correct per-class %), ConfusionMatrixPair (matches
    results.json's matrices exactly), LatencyCostPanel (320ms/4416ms/$9.57,
    honest baseline cost_note instead of a fabricated $0), FailureGallery
    (45/5/5 buckets, real thumbnails, clean evidence quotes), Footer (real
    dataset stats).
- [ ] Record 2–3 min demo video for the CEO email (user's task)
- [x] Restyled to match C10 Labs' aesthetic (2026-08-01)
  - **Looked at the real site first** instead of guessing "bold sans-serif" from
    the user's own unfilled placeholder — screenshotted c10labs.com with
    Playwright, read computed styles (font-family, sizes, letter-spacing,
    colors). Real tokens: headings in Space Grotesk (600 weight, -2.4px tracking
    at 96px), body in Inter, `#fafafa`/`#141414` base, crimson `#e11d48` used
    sparingly (one CTA, one highlighted word per heading, one standout stat).
  - **Resolved the "one accent color" vs "charts need 2 series colors" tension**
    by copying their own convention: their stat row highlights exactly one
    number in crimson and leaves the rest black. Applied directly: baseline =
    ink/near-black (neutral), VLM = crimson (the standout finding). This isn't
    just on-brand — ink/crimson is a stronger accessibility pairing than the
    prior blue/orange (lightness contrast, not just hue, so it survives full
    monochromacy).
  - Achieved almost entirely via **two token files** (`lib/theme.js`,
    `index.css`) — since every component already referenced colors through
    `COLORS.*` / CSS vars rather than hardcoded hex, the full recolor needed
    zero logic changes, exactly matching "don't touch data or logic."
  - Confusion-matrix heatmap ramp switched from blue to grayscale — kept
    quiet/restrained on purpose, reserving crimson for the accent role only.
  - Small structural touches (all visual, no logic): eyebrow-with-rule flourish
    (their signature small detail) on Hero/SectionHeading; two-tone heading in
    InsightBanner (color the word "VLM" crimson, matching their pattern of
    coloring one key word per heading); FailureCard switched from
    white+border+shadow to flat gray card (`bg-[var(--color-card)]`, no
    border) matching their feature-card treatment; correct predictions now
    render as plain text rather than a green pill (more restrained — only the
    disagreement gets flagged); thin full-width dividers between sections
    (`border-black/[0.06]`) replacing pure padding-based separation; bumped
    section padding for more generous whitespace.
  - **Verified with the same real-browser method as the original build**:
    Playwright + Chromium screenshots (not HTTP checks), zero console errors,
    confirmed `bun run lint` and `bun run build` both clean. Playwright removed
    again afterward — not a runtime dependency.

## Review section

### Checkpoint A — scaffold (2026-07-31)
**Done:** Repo structure per TRD §3, `requirements.txt`, `README.md` skeleton,
`run.slurm`, `.gitignore`. No model code, no frontend — as scoped.

**Decisions worth knowing:**
- `run.slurm` is one stage-parameterized template (`sbatch run.slurm data_prep`)
  rather than four near-identical scripts. Guard clauses reject a missing or
  unknown stage before the job burns an allocation.
- Added `data/` and `logs/` — not in the TRD tree, but `data_prep.py` needs an
  input home and SLURM's `--output=logs/...` fails instantly if `logs/` is absent.
- `numpy` pinned `<2.0`: mediapipe wheels still link against numpy 1.x.
- `opencv-python-headless` over `opencv-python`: compute nodes have no GUI libs.
- Checkpoint B deps (`anthropic`, `python-dotenv`) listed now so the HPC conda env
  is built once instead of twice.
- `.gitignore` excludes video and plots but **commits** `metrics.csv` /
  `results.json` — the dashboard reads the latter and the repo should be
  readable without a rerun.

**Verified:** `bash -n run.slurm` passes; both guard clauses tested and exit 2
with the right message. Not verified: SLURM directives (needs an actual submit)
and the pip install (needs the cluster's CUDA).

**Open for user:** confirm partition / GPU constraint / anaconda module on Explorer.

**Next:** `data_prep.py` — source the clips, decide the dataset, build `labels.csv`.
