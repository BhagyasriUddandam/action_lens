#!/usr/bin/env python3
"""VLM action classifier: Claude reads sampled frames and names the action.

The other half of the ActionLens benchmark. Sends a handful of frames per clip
to a vision-language model and asks it to classify the action, logging
predictions, per-clip latency, and actual cost from the API's usage fields.

FAIR FIGHT WITH THE BASELINE:
The prompt is tuned ONLY against the 32 `calib` clips and scored once on the
128 `eval` clips -- the same split baseline_model.py fits its head on. Both
approaches therefore get an equal adaptation budget on identical data.

WHY THE PROMPT ASKS FOR AN ACTION, NOT A POSE:
HMDB51's `sit` and `stand` are transitions (sitting down / standing up) that
are near time-reverses of each other, and the frozen Kinetics-400 backbone in
baseline_model.py cannot separate them -- their class centroids sit at 0.868
cosine similarity. A VLM can reason about direction of motion, so the prompt
states that frames are chronological and makes direction the discriminator.
That contrast is the point of the whole benchmark.

SPENDING REAL MONEY:
Every completed clip is appended to a JSONL cache immediately, so a crash,
rate limit, or Ctrl-C never re-pays for finished work. Re-running resumes.

Run:
    python src/vlm_benchmark.py --limit 4      # cents, sanity check first
    python src/vlm_benchmark.py                # full 160 clips, ~$2-3
    python src/vlm_benchmark.py --split calib  # prompt tuning only
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger("vlm")

MODEL = "claude-opus-5"
FRAMES_PER_CLIP = 8
MAX_WORKERS = 4
MAX_TOKENS = 1024

# claude-opus-5, USD per million tokens. Cache writes bill at 1.25x input,
# cache reads at 0.1x.
PRICE_IN_PER_MTOK = 5.00
PRICE_OUT_PER_MTOK = 25.00
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10

LABELS = ("walking", "sitting", "standing", "falling")


class ActionPrediction(BaseModel):
    """Schema the model is constrained to. Kept small -- output tokens cost 5x
    input, and `evidence` is the only free-text field."""

    action: Literal["walking", "sitting", "standing", "falling"] = Field(
        description="The single action the person performs across these frames."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How certain you are, from 0.0 to 1.0."
    )
    evidence: str = Field(
        max_length=200,
        description="One sentence citing what in the frames decided it, "
        "especially the direction of motion.",
    )


# --- the prompt ---------------------------------------------------------------
# Tuned against the 32 calib clips only. The three "collapse" classes get
# explicit contrastive definitions because sit/stand/fall all involve vertical
# motion and differ only in direction and control.
SYSTEM_PROMPT = """\
You classify a short video clip of a single person into exactly one action.

You will be given frames sampled evenly from one clip, in CHRONOLOGICAL ORDER
and numbered. Judge the ACTION the person performs across the sequence -- not
the pose they happen to hold in any single frame. Direction of motion over the
sequence is usually the deciding evidence.

The four actions:

- walking: the person travels on foot across the scene. Their torso stays
  upright throughout and their position shifts. No sustained vertical
  transition.

- sitting: the person LOWERS themselves into a seated position. Their hips
  descend under control, usually toward a chair, sofa, bed, step, or the
  ground. Early frames show them higher, later frames lower and seated.

- standing: the person RISES to their feet from a seated or crouched position.
  Their hips ascend under control. Early frames show them low or seated, later
  frames show them upright. This is the time-reverse of sitting.

- falling: the person drops to the ground WITHOUT control. Look for loss of
  balance, an unplanned or abrupt descent, limbs flailing or bracing for
  impact, and a final position lying or sprawled on the floor rather than
  seated on a support surface.

The distinctions that matter most:

1. sitting vs standing: both are controlled vertical transitions between a
   seated and an upright pose. They differ ONLY in direction. Compare the
   FIRST frames against the LAST frames. Descending across the sequence means
   sitting; ascending means standing. Do not judge from the middle frames
   alone -- a person mid-transition looks identical either way.

2. falling vs sitting: both descend. Sitting is controlled and ends supported
   on a seat; falling is uncontrolled and ends on the floor.

Choose the single best label even when the clip is dark, blurry, partly
occluded, or the camera moves. Lower your confidence rather than refusing.
"""

USER_INSTRUCTION = (
    "These {n} frames are in chronological order, first to last. "
    "Classify the action."
)


@dataclass
class ClipResult:
    clip_id: str
    label: str
    source: str
    split: str
    pred: str | None
    confidence: float | None
    evidence: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    error: str = ""

    def as_row(self) -> dict:
        return {
            "clip_id": self.clip_id, "label": self.label, "source": self.source,
            "split": self.split, "pred": self.pred, "confidence": self.confidence,
            "correct": self.pred == self.label, "evidence": self.evidence,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost_usd": round(self.cost_usd, 6), "error": self.error,
        }


# --- request construction -----------------------------------------------------


def pick_frames(frames_dir: Path, n: int) -> list[Path]:
    """n frames evenly spaced across the clip, endpoints included.

    The first and last frames carry most of the direction-of-motion signal the
    prompt relies on, so they must always be in the sample.
    """
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    if len(paths) <= n:
        return paths
    step = (len(paths) - 1) / (n - 1)
    return [paths[round(i * step)] for i in range(n)]


def build_content(frame_paths: list[Path]) -> list[dict]:
    """Interleave a label before each image so the model can refer to frame
    order explicitly rather than inferring it from position."""
    content: list[dict] = []
    for i, path in enumerate(frame_paths, 1):
        content.append({"type": "text", "text": f"Frame {i} of {len(frame_paths)}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(path.read_bytes()).decode(),
            },
        })
    content.append({"type": "text", "text": USER_INSTRUCTION.format(n=len(frame_paths))})
    return content


def compute_cost(usage) -> float:
    """USD for one call, from the API's own token counts."""
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        usage.input_tokens * PRICE_IN_PER_MTOK
        + cache_write * PRICE_IN_PER_MTOK * CACHE_WRITE_MULTIPLIER
        + cache_read * PRICE_IN_PER_MTOK * CACHE_READ_MULTIPLIER
        + usage.output_tokens * PRICE_OUT_PER_MTOK
    ) / 1_000_000


# --- one clip -----------------------------------------------------------------


def classify_clip(client, row, cfg) -> ClipResult:
    frame_paths = pick_frames(Path(row.frames_dir), cfg.frames)
    content = build_content(frame_paths)

    base = ClipResult(
        clip_id=row.clip_id, label=row.label, source=row.source, split=row.split,
        pred=None, confidence=None, evidence="", latency_ms=0.0,
        input_tokens=0, output_tokens=0, cache_read_tokens=0,
        cache_write_tokens=0, cost_usd=0.0,
    )

    kwargs = dict(
        model=cfg.model,
        max_tokens=MAX_TOKENS,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            # The system block is identical on every call, so caching it turns
            # ~400 input tokens per clip into a 0.1x cache read.
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": content}],
        output_format=ActionPrediction,
    )
    if cfg.thinking == "disabled":
        # Accepted on claude-opus-5 only at effort `high` or below.
        kwargs["thinking"] = {"type": "disabled"}
    else:
        kwargs["thinking"] = {"type": "adaptive"}
    kwargs["output_config"] = {"effort": cfg.effort}

    t0 = time.perf_counter()
    try:
        response = client.messages.parse(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- recorded per clip, never aborts the run
        base.latency_ms = (time.perf_counter() - t0) * 1000
        base.error = f"{type(exc).__name__}: {exc}"[:300]
        return base
    base.latency_ms = (time.perf_counter() - t0) * 1000

    usage = response.usage
    base.input_tokens = usage.input_tokens
    base.output_tokens = usage.output_tokens
    base.cache_read_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
    base.cache_write_tokens = getattr(usage, "cache_creation_input_tokens", 0) or 0
    base.cost_usd = compute_cost(usage)

    # Safety classifiers can decline a request: HTTP 200, stop_reason "refusal",
    # and no parsed output. Reading the prediction unconditionally would crash.
    if response.stop_reason == "refusal":
        base.error = "refusal"
        return base

    parsed = response.parsed_output
    if parsed is None:
        base.error = f"no parsed output (stop_reason={response.stop_reason})"
        return base

    base.pred = parsed.action
    base.confidence = parsed.confidence
    base.evidence = parsed.evidence
    return base


# --- run ----------------------------------------------------------------------


@dataclass
class Cache:
    """Append-only JSONL of completed clips. Every successful call is flushed
    immediately -- an interrupted run never re-pays for finished work."""

    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        done = {}
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial final line from a hard kill
            if not rec.get("error"):
                done[rec["clip_id"]] = rec
        return done

    def append(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()


def run(df: pd.DataFrame, client, cfg, cache: Cache) -> list[dict]:
    done = cache.load()
    if done:
        log.info("resuming: %d/%d clips already cached", len(done), len(df))

    todo = [r for r in df.itertuples() if r.clip_id not in done]
    rows = [done[r.clip_id] for r in df.itertuples() if r.clip_id in done]
    if not todo:
        return rows

    log.info("classifying %d clips with %s (%d workers)", len(todo), cfg.model, cfg.workers)
    completed = 0
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(classify_clip, client, r, cfg): r for r in todo}
        for fut in as_completed(futures):
            result = fut.result()
            row = result.as_row()
            if not result.error:
                cache.append(row)
            else:
                log.warning("  %s FAILED: %s", result.clip_id, result.error)
            rows.append(row)
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                spent = sum(r["cost_usd"] for r in rows)
                log.info("  %d/%d done | $%.3f spent", completed, len(todo), spent)
    return rows


# --- reporting ----------------------------------------------------------------


def report(df: pd.DataFrame, labels: list[str]) -> dict:
    scored = df[df.pred.notna()]
    if scored.empty:
        log.error("no successful predictions to score")
        return {}

    y_true, y_pred = scored.label.to_numpy(), scored.pred.to_numpy()
    acc = float((y_true == y_pred).mean())

    log.info("\n%s", "=" * 62)
    log.info("VLM RESULTS  (n=%d scored, %d failed)", len(scored), len(df) - len(scored))
    log.info("%s", "=" * 62)
    log.info("  overall accuracy: %.1f%%  (%d/%d)", acc * 100,
             int((y_true == y_pred).sum()), len(y_true))

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    log.info("\n  per-class:")
    log.info("    %-10s %8s %8s %8s %7s", "class", "acc", "prec", "recall", "n")
    per_class = {}
    for c in labels:
        sub = scored[scored.label == c]
        c_acc = float((sub.label == sub.pred).mean()) if len(sub) else float("nan")
        per_class[c] = {
            "accuracy": round(c_acc, 4),
            "precision": round(rep[c]["precision"], 4),
            "recall": round(rep[c]["recall"], 4),
            "f1": round(rep[c]["f1-score"], 4),
            "n": int(rep[c]["support"]),
        }
        log.info("    %-10s %7.1f%% %8.2f %8.2f %7d", c, c_acc * 100,
                 rep[c]["precision"], rep[c]["recall"], int(rep[c]["support"]))

    log.info("\n  confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    log.info("    %-10s %s", "", " ".join(f"{c[:8]:>9}" for c in labels))
    for c, cm_row in zip(labels, cm):
        log.info("    %-10s %s", c, " ".join(f"{v:>9}" for v in cm_row))

    per_source = {}
    fall = scored[scored.label == "falling"]
    if not fall.empty:
        log.info("\n  falling, by source (confound check):")
        for src, g in fall.groupby("source"):
            a = float((g.label == g.pred).mean())
            per_source[src] = {"accuracy": round(a, 4), "n": len(g)}
            log.info("    %-10s %7.1f%%  (%d clips)", src, a * 100, len(g))

    return {
        "overall_accuracy": round(acc, 4), "per_class": per_class,
        "falling_by_source": per_source,
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "n_scored": len(scored), "n_failed": int(len(df) - len(scored)),
    }


def report_cost(df: pd.DataFrame) -> dict:
    total = float(df.cost_usd.sum())
    lat = df[df.latency_ms > 0].latency_ms
    stats = {
        "total_usd": round(total, 4),
        "per_clip_usd": round(total / max(len(df), 1), 5),
        "per_1k_clips_usd": round(total / max(len(df), 1) * 1000, 2),
        "input_tokens": int(df.input_tokens.sum()),
        "output_tokens": int(df.output_tokens.sum()),
        "cache_read_tokens": int(df.cache_read_tokens.sum()),
        "cache_write_tokens": int(df.cache_write_tokens.sum()),
        "latency_mean_ms": round(float(lat.mean()), 1) if len(lat) else 0.0,
        "latency_median_ms": round(float(lat.median()), 1) if len(lat) else 0.0,
        "latency_p90_ms": round(float(lat.quantile(0.9)), 1) if len(lat) else 0.0,
    }
    log.info("\n  latency per clip: mean %.0f ms | median %.0f ms | p90 %.0f ms",
             stats["latency_mean_ms"], stats["latency_median_ms"], stats["latency_p90_ms"])
    log.info("  cost: $%.4f total | $%.5f per clip | ~$%.2f per 1k clips",
             stats["total_usd"], stats["per_clip_usd"], stats["per_1k_clips_usd"])
    cached = stats["cache_read_tokens"]
    if cached:
        log.info("  prompt cache: %d tokens read at 0.1x (%d written)",
                 cached, stats["cache_write_tokens"])
    return stats


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--out", type=Path, default=Path("results/vlm_predictions.csv"))
    p.add_argument("--metrics-out", type=Path, default=Path("results/vlm_metrics.json"))
    p.add_argument("--cache", type=Path, default=Path("results/vlm_cache.jsonl"),
                   help="append-only completed-clip cache; delete to force a rerun")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--frames", type=int, default=FRAMES_PER_CLIP,
                   help="frames sent per clip (of the 16 on disk)")
    p.add_argument("--split", default="eval", choices=["eval", "calib", "all"],
                   help="eval = the scored benchmark; calib = the 32 prompt-tuning clips")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                   help="thinking depth; low suits a bounded classification task")
    p.add_argument("--thinking", default="adaptive", choices=["adaptive", "disabled"])
    p.add_argument("--limit", type=int, default=None, help="first N clips only (cheap check)")
    p.add_argument("--dry-run", action="store_true",
                   help="build every request and exercise the full pipeline "
                        "with a deterministic stub instead of calling the API")
    cfg = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not cfg.labels.is_file():
        log.error("labels file not found: %s\nRun: python src/data_prep.py", cfg.labels)
        return 1
    df = pd.read_csv(cfg.labels)
    if cfg.split != "all":
        df = df[df.split == cfg.split]
    if cfg.limit:
        # Stratified so a cheap smoke test still exercises all four classes --
        # head() would spend the whole budget on whichever label sorts first.
        per = max(1, cfg.limit // df.label.nunique())
        df = df.groupby("label", group_keys=False).head(per)
    df = df.reset_index(drop=True)
    if df.empty:
        log.error("no clips selected")
        return 1

    if cfg.dry_run:
        from tests.stub_client import StubClient  # noqa: PLC0415 -- test-only import

        client = StubClient()
        cache = Cache(Path("/tmp/vlm_dryrun_cache.jsonl"))
        cache.path.unlink(missing_ok=True)
        log.info("DRY RUN -- no API calls, stubbed responses")
    else:
        import anthropic  # noqa: PLC0415 -- keeps --dry-run importable without a key

        client = anthropic.Anthropic()
        cache = Cache(cfg.cache)

    log.info("split=%s | clips=%d | frames/clip=%d | effort=%s | thinking=%s",
             cfg.split, len(df), cfg.frames, cfg.effort, cfg.thinking)

    rows = run(df, client, cfg, cache)
    out = pd.DataFrame(rows).sort_values(["label", "source", "clip_id"]).reset_index(drop=True)

    # Always the full class list, so the confusion matrix keeps its 4x4 shape
    # even on a subset run.
    metrics = report(out, list(LABELS))
    metrics["cost"] = report_cost(out)
    metrics.update({"model": cfg.model, "frames_per_clip": cfg.frames,
                    "effort": cfg.effort, "thinking": cfg.thinking, "split": cfg.split})

    if not cfg.dry_run:
        cfg.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cfg.out, index=False)
        cfg.metrics_out.write_text(json.dumps(metrics, indent=2))
        log.info("\nwrote %s and %s", cfg.out, cfg.metrics_out)
    else:
        log.info("\n(dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
