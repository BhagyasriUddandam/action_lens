#!/usr/bin/env python3
"""Build the ActionLens 4-class clip dataset from HMDB51 + URFD.

Reads the raw downloads produced by scripts/fetch_hmdb51.sh and
scripts/fetch_urfd.sh, and normalises both into one format:

    data/frames/{label}/{clip_id}/frame_00.jpg ... frame_15.jpg
    data/labels.csv

Why two sources for `falling`: if every falling clip came from URFD, the
source would be a perfect proxy for the label and a model could score well by
recognising URFD's fixed camera and room rather than recognising a fall.
Mixing in HMDB51 fall_floor breaks that confound, and the `source` column
lets evaluate.py report staged-vs-real fall accuracy separately.

Run:
    python src/data_prep.py                 # full build
    python src/data_prep.py --limit 3       # quick smoke test
    python src/data_prep.py --force         # regenerate existing frames
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import pandas as pd

log = logging.getLogger("data_prep")

# --- dataset definition ------------------------------------------------------
# (source, source_class, n_clips) per canonical label.
CLASS_SOURCES: dict[str, list[tuple[str, str | None, int]]] = {
    "walking": [("hmdb51", "walk", 40)],
    "sitting": [("hmdb51", "sit", 40)],
    "standing": [("hmdb51", "stand", 40)],
    "falling": [("urfd", None, 25), ("hmdb51", "fall_floor", 15)],
}

FRAMES_PER_CLIP = 16
RESIZE_SHORT_SIDE = 256
JPEG_QUALITY = 90
SEED = 42
CALIB_FRAC = 0.2

# URFD frames to keep either side of the annotated fall, so the window covers
# the run-up and the landing rather than only the transition itself.
URFD_PAD = 8
URFD_LABELS_CSV = "urfall-cam0-falls.csv"

# HMDB51 encodes video quality in the filename (..._goo_/_med_/_bad_).
# `bad` clips are dark, heavily compressed or motion-blurred. Excluding them
# still leaves 92-496 clips per class, far above the 40 we need.
QUALITY_RE = re.compile(r"_(goo|med|bad)_")
EXCLUDE_QUALITY = ("bad",)

# Many HMDB51 clips are cut from films and span a shot boundary, so some
# sampled frames show a completely different scene (measured: 18-38% of clips
# per class, worst in fall_floor).
#
# Detection is on mean absolute pixel difference, NOT histogram correlation.
# Histograms were tried first and missed real cuts: an Oceans12 walk clip cuts
# from a building exterior to a night street, but both scenes are warm-toned so
# HSV correlation stayed at 0.86 -- nowhere near a 0.5 threshold. The same cut
# shows a pixel diff of 55.6 against a 0.6-10.2 baseline for the rest of the clip.
#
# The threshold is adaptive because a clip's baseline diff depends on how much
# motion it contains: a pair counts as a cut only if it exceeds both an absolute
# floor and a multiple of that clip's own median diff. Measured on 200 clips,
# non-cut pairs sit at ratio ~1.5-2.4 (URFD's fixed camera peaks at 2.5), so 4.0
# leaves clear headroom while still catching the 11x Oceans12 cut.
CUT_ABS_FLOOR = 30.0
CUT_RATIO = 4.0
MAX_CUTS = 0


@dataclass
class Candidate:
    """A clip we might use. Pixels are only decoded if it reaches acceptance."""

    clip_id: str
    label: str
    source: str
    source_path: Path
    src_frame_count: int
    fps_orig: float
    loader: Callable[[], list[np.ndarray]] = field(repr=False)
    src_width: int = 0
    src_height: int = 0
    # Dimensions of the frames actually written. Always read back from disk so
    # a cached re-run reports the same numbers as a fresh extract.
    width: int = 0
    height: int = 0
    split: str = "eval"

    @property
    def duration_s(self) -> float:
        return round(self.src_frame_count / self.fps_orig, 3) if self.fps_orig else 0.0


# --- shared frame ops (source-agnostic) --------------------------------------


def sample_uniform(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Pick n frames evenly across the clip, endpoints included."""
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
    return [frames[i] for i in idx]


def resize_short_side(img: np.ndarray, target: int) -> np.ndarray:
    """Scale so the shorter side == target, preserving aspect ratio.

    HMDB51 clip widths vary (320/416/592 at height 240), so this is required,
    not cosmetic. Downscaling uses INTER_AREA, which avoids the aliasing
    INTER_LINEAR produces.
    """
    h, w = img.shape[:2]
    if min(h, w) == target:
        return img
    scale = target / min(h, w)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def count_hard_cuts(frames: list[np.ndarray], abs_floor: float, ratio: float) -> int:
    """Number of shot boundaries among consecutive frames.

    Run on the SAMPLED frames rather than the source clip, because those are
    the frames that actually end up in the dataset -- a cut we never sampled
    does not matter. See CUT_ABS_FLOOR for why this uses pixel difference
    rather than histogram correlation.
    """
    if len(frames) < 3:
        return 0
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 64)) for f in frames]
    diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(small, small[1:])]
    # Median over the clip's own pairs = its motion baseline. Guarded against
    # zero so a perfectly static clip cannot make every pair look like a cut.
    baseline = max(float(np.median(diffs)), 1e-6)
    return sum(1 for d in diffs if d > abs_floor and d > ratio * baseline)


def write_frames(out_dir: Path, frames: list[np.ndarray], quality: int) -> None:
    """Write frames as frame_NN.jpg.

    Writes to a temp dir then renames, so an interrupted run never leaves a
    half-populated dir that a later run would mistake for complete.
    """
    tmp_dir = out_dir.with_name(out_dir.name + ".partial")
    if tmp_dir.exists():
        for f in tmp_dir.iterdir():
            f.unlink()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        path = tmp_dir / f"frame_{i:02d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError(f"failed to write {path}")

    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
        out_dir.rmdir()
    tmp_dir.rename(out_dir)


def is_complete(out_dir: Path, n_frames: int) -> bool:
    return out_dir.is_dir() and len(list(out_dir.glob("frame_*.jpg"))) == n_frames


def read_output_dims(out_dir: Path) -> tuple[int, int]:
    """(width, height) of the written frames, read back from disk.

    Both the fresh-extract and the reuse path go through here, so labels.csv
    cannot report post-resize dimensions on one run and source dimensions on
    the next.
    """
    first = out_dir / "frame_00.jpg"
    img = cv2.imread(str(first))
    if img is None:
        raise OSError(f"cannot read back written frame: {first}")
    return img.shape[1], img.shape[0]


# --- source adapter: HMDB51 (.avi via OpenCV) --------------------------------


def _read_avi(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def probe_avi(path: Path) -> tuple[int, float, int, int] | None:
    """(frame_count, fps, width, height) without decoding the whole clip."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return (n, fps, w, h) if n > 0 else None


def hmdb51_candidates(
    raw_root: Path, label: str, source_class: str, rng: random.Random,
    frames_per_clip: int, exclude_quality: tuple[str, ...],
) -> Iterator[Candidate]:
    class_dir = raw_root / "hmdb51" / source_class
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"HMDB51 class dir not found: {class_dir}\nRun: ./scripts/fetch_hmdb51.sh"
        )
    paths = sorted(class_dir.glob("*.avi"))
    rng.shuffle(paths)

    for path in paths:
        m = QUALITY_RE.search(path.name)
        if m and m.group(1) in exclude_quality:
            continue
        probed = probe_avi(path)
        if probed is None:
            log.warning("    skip (unreadable): %s", path.name)
            continue
        count, fps, w, h = probed
        if count < frames_per_clip:
            log.warning("    skip (%d < %d frames): %s", count, frames_per_clip, path.name)
            continue
        yield Candidate(
            clip_id=f"hmdb51_{path.stem}",
            label=label,
            source="hmdb51",
            source_path=path,
            src_frame_count=count,
            fps_orig=fps,
            src_width=w,
            src_height=h,
            loader=lambda p=path: _read_avi(p),
        )


# --- source adapter: URFD (PNG sequences + aggregated label csv) -------------


def load_urfd_labels(raw_root: Path) -> pd.DataFrame:
    """Per-frame pose labels for all 30 fall sequences.

    NOTE: these live in ONE aggregated file (urfall-cam0-falls.csv), not in the
    per-sequence fall-NN-data.csv -- that one is a camera/accelerometer sync log
    with no pose labels. No header; label is -1 not-lying, 0 falling, 1 lying.
    """
    path = raw_root / "urfd" / URFD_LABELS_CSV
    if not path.is_file():
        raise FileNotFoundError(
            f"URFD label file not found: {path}\nRun: ./scripts/fetch_urfd.sh"
        )
    return pd.read_csv(path, header=None, usecols=[0, 1, 2], names=["seq", "frame", "label"])


def urfd_fall_window(labels: pd.DataFrame, seq: str, pad: int) -> tuple[int, int] | None:
    """Frame range covering the fall, as (start, end) inclusive.

    A URFD sequence is mostly NOT falling (walk in -> stand -> fall -> lie
    still), so sampling the whole sequence would label mostly-upright frames
    as `falling`. Window from the first `falling` frame to the first `lying`
    frame, padded either side.
    """
    g = labels[labels.seq == seq]
    if g.empty:
        return None
    falling = g.loc[g.label == 0, "frame"]
    lying = g.loc[g.label == 1, "frame"]
    if falling.empty:
        return None
    start = int(falling.min())
    end = int(lying.min()) if not lying.empty else int(falling.max())
    lo, hi = int(g.frame.min()), int(g.frame.max())
    return max(lo, start - pad), min(hi, end + pad)


def _urfd_frame_dir(seq_dir: Path) -> Path | None:
    """The zips extract one level deeper than the zip name
    (fall-01-cam0-rgb/fall-01-cam0-rgb/*.png), so resolve whichever level
    actually holds the PNGs."""
    if any(seq_dir.glob("*.png")):
        return seq_dir
    nested = seq_dir / seq_dir.name
    if nested.is_dir() and any(nested.glob("*.png")):
        return nested
    for sub in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
        if any(sub.glob("*.png")):
            return sub
    return None


def _read_urfd_window(frame_dir: Path, start: int, end: int) -> list[np.ndarray]:
    """Read PNGs whose 1-based frame index falls inside [start, end].

    Filenames are fall-NN-cam0-rgb-{frame:03d}.png and the index matches the
    label csv's frame column exactly (verified against fall-01: 160 labels,
    160 PNGs, transition at 83 -> 113).
    """
    frames = []
    for path in sorted(frame_dir.glob("*.png")):
        try:
            idx = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if start <= idx <= end:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)
    return frames


def urfd_candidates(
    raw_root: Path, label: str, rng: random.Random, frames_per_clip: int, pad: int,
) -> Iterator[Candidate]:
    urfd_root = raw_root / "urfd"
    if not urfd_root.is_dir():
        raise FileNotFoundError(
            f"URFD dir not found: {urfd_root}\nRun: ./scripts/fetch_urfd.sh"
        )
    seq_dirs = sorted(p for p in urfd_root.glob("fall-*-cam0-rgb") if p.is_dir())
    rng.shuffle(seq_dirs)
    labels = load_urfd_labels(raw_root)

    for seq_dir in seq_dirs:
        seq = seq_dir.name.replace("-cam0-rgb", "")
        frame_dir = _urfd_frame_dir(seq_dir)
        if frame_dir is None:
            log.warning("    skip (no PNGs): %s", seq_dir.name)
            continue
        window = urfd_fall_window(labels, seq, pad)
        if window is None:
            log.warning("    skip (no fall annotation): %s", seq)
            continue
        start, end = window
        n_window = end - start + 1
        if n_window < frames_per_clip:
            log.warning("    skip (window %d < %d frames): %s", n_window, frames_per_clip, seq)
            continue

        first_png = next(iter(sorted(frame_dir.glob("*.png"))), None)
        sample = cv2.imread(str(first_png)) if first_png else None
        h, w = sample.shape[:2] if sample is not None else (0, 0)

        yield Candidate(
            clip_id=f"urfd_{seq}",
            label=label,
            source="urfd",
            source_path=seq_dir,
            src_frame_count=n_window,
            fps_orig=30.0,  # URFD cam0 records at 30 fps
            src_width=w,
            src_height=h,
            loader=lambda d=frame_dir, s=start, e=end: _read_urfd_window(d, s, e),
        )


# --- build -------------------------------------------------------------------


@dataclass
class BuildStats:
    written: int = 0
    reused: int = 0
    rejected_cuts: int = 0
    rejected_decode: int = 0


def fill_source(
    candidates: Iterator[Candidate], target: int, out_root: Path, cfg: argparse.Namespace,
    stats: BuildStats,
) -> list[Candidate]:
    """Draw candidates until `target` are accepted and written.

    Rejection (short decode, shot boundary) pulls the next candidate rather
    than shrinking the class, so a filtered class still reaches its count.
    Selection is deterministic for a given seed, so a re-run accepts exactly
    the same clips -- which is what makes the skip-if-present path safe.
    """
    accepted: list[Candidate] = []
    for cand in candidates:
        if len(accepted) >= target:
            break
        out_dir = out_root / "frames" / cand.label / cand.clip_id

        if not cfg.force and is_complete(out_dir, cfg.frames):
            stats.reused += 1
        else:
            frames = cand.loader()
            if len(frames) < cfg.frames:
                log.warning("    reject (decoded %d frames): %s", len(frames), cand.clip_id)
                stats.rejected_decode += 1
                continue

            sampled = sample_uniform(frames, cfg.frames)
            if cfg.max_cuts >= 0:
                n_cuts = count_hard_cuts(sampled, cfg.cut_floor, cfg.cut_ratio)
                if n_cuts > cfg.max_cuts:
                    log.info("    reject (%d shot cut(s)): %s", n_cuts, cand.clip_id)
                    stats.rejected_cuts += 1
                    continue

            write_frames(out_dir, [resize_short_side(f, cfg.size) for f in sampled], cfg.quality)
            stats.written += 1

        cand.width, cand.height = read_output_dims(out_dir)
        accepted.append(cand)

    if len(accepted) < target:
        log.warning("    only %d of %d requested clips available", len(accepted), target)
    return accepted


def build(cfg: argparse.Namespace) -> tuple[list[Candidate], BuildStats]:
    stats = BuildStats()
    accepted: list[Candidate] = []

    for label, specs in CLASS_SOURCES.items():
        log.info("%s", label)
        # Per-label RNG so changing one class's count cannot reshuffle another.
        rng = random.Random(f"{cfg.seed}:{label}")
        for source, source_class, default_n in specs:
            target = default_n
            if cfg.per_class is not None:
                # Preserve each source's share of the class when rescaling.
                share = default_n / sum(s[2] for s in specs)
                target = max(1, round(cfg.per_class * share))
            if cfg.limit is not None:
                target = min(target, cfg.limit)

            if source == "hmdb51":
                stream = hmdb51_candidates(
                    cfg.raw_root, label, source_class, rng, cfg.frames, cfg.exclude_quality
                )
            elif source == "urfd":
                stream = urfd_candidates(cfg.raw_root, label, rng, cfg.frames, cfg.urfd_pad)
            else:
                raise ValueError(f"unknown source: {source}")

            got = fill_source(stream, target, cfg.out_root, cfg, stats)
            log.info("  %-8s %d/%d clips", source, len(got), target)
            accepted.extend(got)

    return accepted, stats


def assign_splits(clips: list[Candidate], calib_frac: float, seed: int) -> None:
    """Stratified eval/calib split.

    Nothing is trained here, so this is not a train split: `calib` is a small
    held-out set for tuning the VLM prompt in Checkpoint B. Tuning on the same
    clips we report accuracy over would inflate the headline number.
    Stratified by (label, source) so the falling class's URFD/HMDB51 mix is
    preserved on both sides.
    """
    by_stratum: dict[tuple[str, str], list[Candidate]] = {}
    for c in clips:
        by_stratum.setdefault((c.label, c.source), []).append(c)

    for (label, source), group in sorted(by_stratum.items()):
        group.sort(key=lambda c: c.clip_id)
        random.Random(f"{seed}:split:{label}:{source}").shuffle(group)
        n_calib = round(len(group) * calib_frac)
        for c in group[:n_calib]:
            c.split = "calib"
        for c in group[n_calib:]:
            c.split = "eval"


def to_dataframe(clips: list[Candidate], out_root: Path, n_frames: int) -> pd.DataFrame:
    rows = [
        {
            "clip_id": c.clip_id,
            "label": c.label,
            "source": c.source,
            "split": c.split,
            "n_frames": n_frames,
            "src_frame_count": c.src_frame_count,
            "fps_orig": round(c.fps_orig, 2),
            "duration_s": c.duration_s,
            "width": c.width,
            "height": c.height,
            "src_width": c.src_width,
            "src_height": c.src_height,
            "frames_dir": str(out_root / "frames" / c.label / c.clip_id),
            "source_path": str(c.source_path),
        }
        for c in clips
    ]
    return pd.DataFrame(rows).sort_values(["label", "source", "clip_id"]).reset_index(drop=True)


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--out-root", type=Path, default=Path("data"))
    p.add_argument("--per-class", type=int, default=None,
                   help="override clips per class (default: 40, split across sources)")
    p.add_argument("--frames", type=int, default=FRAMES_PER_CLIP)
    p.add_argument("--size", type=int, default=RESIZE_SHORT_SIDE, help="target shorter side")
    p.add_argument("--quality", type=int, default=JPEG_QUALITY)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--calib-frac", type=float, default=CALIB_FRAC)
    p.add_argument("--urfd-pad", type=int, default=URFD_PAD)
    p.add_argument("--limit", type=int, default=None, help="cap clips per source (smoke test)")
    p.add_argument("--force", action="store_true", help="re-extract clips that already exist")
    p.add_argument("--keep-bad-quality", action="store_true",
                   help="keep HMDB51 clips tagged _bad_ (excluded by default)")
    p.add_argument("--max-cuts", type=int, default=MAX_CUTS,
                   help="max shot boundaries allowed in the sampled frames; -1 disables the check")
    p.add_argument("--cut-floor", type=float, default=CUT_ABS_FLOOR,
                   help="minimum mean pixel difference for a pair to count as a cut")
    p.add_argument("--cut-ratio", type=float, default=CUT_RATIO,
                   help="multiple of the clip's own median diff that counts as a cut")
    cfg = p.parse_args(argv)
    cfg.exclude_quality = () if cfg.keep_bad_quality else EXCLUDE_QUALITY

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    clips, stats = build(cfg)
    if not clips:
        log.error("no clips selected -- did the fetch scripts run?")
        return 1

    assign_splits(clips, cfg.calib_frac, cfg.seed)
    df = to_dataframe(clips, cfg.out_root, cfg.frames)

    csv_path = cfg.out_root / "labels.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    log.info("\n%s", "=" * 60)
    log.info("wrote %s", csv_path)
    log.info("  %d extracted, %d reused, %d rejected (%d shot cuts, %d short decode)",
             stats.written, stats.reused, stats.rejected_cuts + stats.rejected_decode,
             stats.rejected_cuts, stats.rejected_decode)
    log.info("%s", "=" * 60)
    counts = df.groupby(["label", "source"]).size().reset_index(name="clips")
    for label in CLASS_SOURCES:
        sub = counts[counts.label == label]
        detail = ", ".join(f"{r.source}={r.clips}" for r in sub.itertuples())
        log.info("  %-9s %3d   (%s)", label, int(sub.clips.sum()), detail)
    log.info("  %-9s %3d", "TOTAL", len(df))
    log.info("  splits: %s", df.split.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
