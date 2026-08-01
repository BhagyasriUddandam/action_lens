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
  - [ ] Verify: labels.csv has 160 rows, 40/class, both sources present in `falling`
- [ ] baseline_model.py: run a PRETRAINED pose/action model, output per-clip predictions + accuracy
- [ ] Verify: pipeline runs end-to-end on HPC, prints accuracy
- [ ] Write README setup steps (yourself)

## CHECKPOINT B — VLM Benchmark (full C10 workflow)
- [ ] vlm_benchmark.py: send sampled frames to a VLM, get action labels, log latency + approx cost
- [ ] evaluate.py: baseline vs VLM — accuracy, per-class, failure cases, latency/cost table + plots
- [ ] Save results/metrics.csv and results/results.json
- [ ] Write FINDINGS.md (yourself): when to use which approach

## CHECKPOINT C — Dashboard (CEO wow)
- [ ] Scaffold React+Vite app with bun, Tailwind, GSAP (current Node LTS)
- [ ] Components (SOLID): ClipViewer, PredictionCompare, MetricsCharts, InsightBanner
- [ ] Load results.json; animate charts with GSAP/Animate UI
- [ ] Verify it renders and looks polished
- [ ] Record 2–3 min demo video for the CEO email

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
