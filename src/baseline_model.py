#!/usr/bin/env python3
"""Baseline action classifier: frozen pretrained video backbone + small head.

Reads the clips built by data_prep.py and produces per-clip predictions plus
accuracy for the four ActionLens classes.

WHY NOT USE THE PRETRAINED CLASSIFIER HEAD DIRECTLY:
Every torchvision video model is pretrained on Kinetics-400, and K400 contains
none of our classes -- no `standing`, no `falling`, and no generic `walking`
or `sitting` (only "walking the dog" and "situp"). Mapping K400 logits onto
our labels would score near zero on three classes out of four.

So the backbone is used as a FROZEN FEATURE EXTRACTOR and only a 4-class head
is fitted, on the 32 `calib` clips. Nothing is trained from scratch and the
backbone's weights are never updated -- this is a linear probe.

The `calib`/`eval` split matters for the Checkpoint B comparison: the VLM gets
the same 32 calib clips to tune its prompt against and is scored on the same
128 eval clips, so both approaches get an equal adaptation budget.

Run:
    python src/baseline_model.py                     # default mvit_v2_s
    python src/baseline_model.py --model r3d_18      # faster, weaker features
    python src/baseline_model.py --limit 20          # smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.io import decode_image
from torchvision.models import video as tv_video

log = logging.getLogger("baseline")

# Backbones are ranked by published Kinetics-400 top-1. Picking the default on
# published accuracy is an a-priori criterion -- it never touches our eval set,
# unlike picking whichever scores best on our own benchmark.
BACKBONES = {
    "mvit_v2_s": ("MViT_V2_S_Weights", "head"),      # K400 acc@1 80.8
    "r2plus1d_18": ("R2Plus1D_18_Weights", "fc"),    # K400 acc@1 67.5
    "r3d_18": ("R3D_18_Weights", "fc"),              # K400 acc@1 63.2
}
DEFAULT_MODEL = "mvit_v2_s"
LATENCY_SAMPLE = 20


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- feature extraction ------------------------------------------------------


@dataclass
class Backbone:
    model: nn.Module
    transform: object
    dim: int
    name: str
    n_params: int


def load_backbone(name: str, device: torch.device) -> Backbone:
    """Pretrained video model with its classifier head removed.

    Returns pooled embeddings rather than K400 logits. Frozen: eval mode and
    requires_grad_(False) everywhere.
    """
    if name not in BACKBONES:
        raise ValueError(f"unknown model {name!r}; choose from {list(BACKBONES)}")
    weights_attr, head_attr = BACKBONES[name]
    weights = getattr(tv_video, weights_attr).KINETICS400_V1
    model = getattr(tv_video, name)(weights=weights)

    head = getattr(model, head_attr)
    # mvit's head is Sequential(Dropout, Linear); the resnets' is a bare Linear.
    linear = head[-1] if isinstance(head, nn.Sequential) else head
    dim = linear.in_features
    setattr(model, head_attr, nn.Identity())

    model.eval().requires_grad_(False).to(device)
    return Backbone(
        model=model,
        transform=weights.transforms(),
        dim=dim,
        name=name,
        n_params=sum(p.numel() for p in model.parameters()),
    )


def load_clip(frames_dir: Path) -> torch.Tensor:
    """Read a clip's frames as a uint8 tensor [T, C, H, W]."""
    paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frames_dir}")
    return torch.stack([decode_image(str(p)) for p in paths])


@torch.inference_mode()
def extract_features(
    df: pd.DataFrame, bb: Backbone, device: torch.device, batch_size: int
) -> np.ndarray:
    """Embed every clip. Batched for throughput; latency is measured separately."""
    feats: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        out = bb.model(x)
        feats.append(out.float().cpu().numpy())
        batch.clear()

    for i, row in enumerate(df.itertuples(), 1):
        clip = bb.transform(load_clip(Path(row.frames_dir)))
        batch.append(clip)
        if len(batch) == batch_size:
            flush()
            log.info("  embedded %d/%d", i, len(df))
    flush()
    log.info("  embedded %d/%d", len(df), len(df))
    return np.concatenate(feats, axis=0)


@torch.inference_mode()
def measure_latency(
    df: pd.DataFrame, bb: Backbone, device: torch.device, n: int
) -> dict[str, float]:
    """Per-clip inference time at batch=1.

    Batch=1 is the number worth comparing against a per-clip VLM API call.
    Excludes disk read and preprocessing so it measures the model, and runs a
    warmup first because the first call on CUDA/MPS pays lazy init.
    """
    sample = df.head(n)
    clips = [bb.transform(load_clip(Path(r.frames_dir))).unsqueeze(0).to(device)
             for r in sample.itertuples()]
    if not clips:
        return {}

    bb.model(clips[0])  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    times = []
    for clip in clips:
        t0 = time.perf_counter()
        bb.model(clip)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "n_sampled": len(arr),
        "mean_ms": round(float(arr.mean()), 1),
        "median_ms": round(float(np.median(arr)), 1),
        "p90_ms": round(float(np.percentile(arr, 90)), 1),
    }


# --- classifier heads --------------------------------------------------------


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class NearestCentroid:
    """Predict the class whose mean embedding is closest (cosine).

    Standard few-shot baseline. With 8 examples per class it is far more stable
    than a discriminative fit, and it has no hyperparameters to tune.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        xn = l2_normalize(x)
        self.classes_ = np.unique(y)
        self.centroids_ = l2_normalize(
            np.stack([xn[y == c].mean(axis=0) for c in self.classes_])
        )
        return self

    def decision(self, x: np.ndarray) -> np.ndarray:
        return l2_normalize(x) @ self.centroids_.T

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.decision(x).argmax(axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Softmax over cosine similarities, temperature-scaled so the numbers
        # are readable as confidences. Not calibrated -- for ranking only.
        s = self.decision(x) * 10.0
        e = np.exp(s - s.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def build_classifier(kind: str):
    if kind == "centroid":
        return NearestCentroid()
    if kind == "logreg":
        # multinomial is the default and `multi_class` was removed in sklearn 1.9.
        return LogisticRegression(max_iter=2000, C=1.0)
    raise ValueError(f"unknown classifier {kind!r}")


def loo_cv_accuracy(kind: str, x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out accuracy on the calib set.

    Classifier choice is made here rather than on the eval split -- selecting a
    head by its eval score would make the reported accuracy optimistic.
    """
    correct = 0
    for i in range(len(x)):
        mask = np.ones(len(x), dtype=bool)
        mask[i] = False
        # LOO can leave a class with too few examples for logreg; skip those folds.
        if len(np.unique(y[mask])) < len(np.unique(y)):
            continue
        clf = build_classifier(kind).fit(x[mask], y[mask])
        correct += int(clf.predict(x[i : i + 1])[0] == y[i])
    return correct / len(x)


# --- reporting ---------------------------------------------------------------


def report(df_eval: pd.DataFrame, labels: list[str]) -> dict:
    y_true = df_eval.label.to_numpy()
    y_pred = df_eval.pred.to_numpy()
    acc = float((y_true == y_pred).mean())

    log.info("\n%s", "=" * 62)
    log.info("BASELINE RESULTS  (eval split, n=%d)", len(df_eval))
    log.info("%s", "=" * 62)
    log.info("  overall accuracy: %.1f%%  (%d/%d)", acc * 100,
             int((y_true == y_pred).sum()), len(y_true))

    log.info("\n  per-class:")
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    log.info("    %-10s %8s %8s %8s %7s", "class", "acc", "prec", "recall", "n")
    per_class = {}
    for c in labels:
        sub = df_eval[df_eval.label == c]
        c_acc = float((sub.label == sub.pred).mean()) if len(sub) else float("nan")
        per_class[c] = {
            "accuracy": round(c_acc, 4),
            "precision": round(rep[c]["precision"], 4),
            "recall": round(rep[c]["recall"], 4),
            "f1": round(rep[c]["f1-score"], 4),
            "n": int(rep[c]["support"]),
        }
        log.info("    %-10s %7.1f%% %8.2f %8.2f %7d",
                 c, c_acc * 100, rep[c]["precision"], rep[c]["recall"], int(rep[c]["support"]))

    log.info("\n  confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    log.info("    %-10s %s", "", " ".join(f"{c[:8]:>9}" for c in labels))
    for c, row in zip(labels, cm):
        log.info("    %-10s %s", c, " ".join(f"{v:>9}" for v in row))

    # The reason `falling` draws on two datasets: if it came only from URFD,
    # a model could win by recognising the room instead of the fall.
    per_source = {}
    fall = df_eval[df_eval.label == "falling"]
    if not fall.empty:
        log.info("\n  falling, by source (confound check):")
        for src, g in fall.groupby("source"):
            a = float((g.label == g.pred).mean())
            per_source[src] = {"accuracy": round(a, 4), "n": len(g)}
            log.info("    %-10s %7.1f%%  (%d clips)", src, a * 100, len(g))

    return {"overall_accuracy": round(acc, 4), "per_class": per_class,
            "falling_by_source": per_source,
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()}}


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    p.add_argument("--out", type=Path, default=Path("results/baseline_predictions.csv"))
    p.add_argument("--metrics-out", type=Path, default=Path("results/baseline_metrics.json"))
    p.add_argument("--model", default=DEFAULT_MODEL, choices=list(BACKBONES))
    p.add_argument("--classifier", default="auto", choices=["auto", "centroid", "logreg"],
                   help="auto selects by leave-one-out CV on the calib split")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="use only N clips (smoke test)")
    p.add_argument("--features-out", type=Path, default=None,
                   help="optional .npz dump of embeddings for reuse")
    p.add_argument("--features-in", type=Path, default=None,
                   help="reuse embeddings from a previous --features-out (skips the backbone)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if not args.labels.is_file():
        log.error("labels file not found: %s\nRun: python src/data_prep.py", args.labels)
        return 1
    df = pd.read_csv(args.labels)
    if args.limit:
        df = df.groupby("label", group_keys=False).head(max(2, args.limit // df.label.nunique()))
    df = df.reset_index(drop=True)

    device = pick_device(args.device)
    log.info("device: %s | clips: %d", device, len(df))

    bb = load_backbone(args.model, device)
    log.info("backbone: %s (%.1fM params, %d-d features) -- FROZEN, head removed",
             bb.name, bb.n_params / 1e6, bb.dim)

    if args.features_in and args.features_in.is_file():
        cached = np.load(args.features_in, allow_pickle=True)
        if list(cached["clip_id"]) != list(df.clip_id):
            log.error("cached features in %s do not match %s -- re-extract",
                      args.features_in, args.labels)
            return 1
        feats = cached["features"]
        log.info("reusing cached features from %s", args.features_in)
    else:
        log.info("extracting features...")
        t0 = time.perf_counter()
        feats = extract_features(df, bb, device, args.batch_size)
        log.info("  done in %.1fs", time.perf_counter() - t0)

    if args.features_out:
        args.features_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.features_out, features=feats, clip_id=df.clip_id.to_numpy())

    calib = df.split == "calib"
    evalm = df.split == "eval"
    if not calib.any():
        log.error("no calib clips -- cannot fit a head")
        return 1
    x_cal, y_cal = feats[calib.to_numpy()], df.loc[calib, "label"].to_numpy()

    if args.classifier == "auto":
        log.info("\nselecting head by leave-one-out CV on %d calib clips:", len(x_cal))
        scores = {k: loo_cv_accuracy(k, x_cal, y_cal) for k in ("centroid", "logreg")}
        for k, v in scores.items():
            log.info("  %-9s LOO acc %.1f%%", k, v * 100)
        kind = max(scores, key=scores.get)
        log.info("  -> using %s", kind)
    else:
        kind = args.classifier

    clf = build_classifier(kind).fit(x_cal, y_cal)

    proba = clf.predict_proba(feats)
    classes = list(clf.classes_)
    df["pred"] = [classes[i] for i in proba.argmax(axis=1)]
    df["confidence"] = proba.max(axis=1).round(4)
    df["correct"] = df.pred == df.label

    lat = measure_latency(df, bb, device, LATENCY_SAMPLE)
    labels = sorted(df.label.unique())
    metrics = report(df[evalm], labels)

    log.info("\n  latency (batch=1, %s, n=%d): mean %.1f ms | median %.1f ms | p90 %.1f ms",
             device.type, lat.get("n_sampled", 0), lat.get("mean_ms", 0),
             lat.get("median_ms", 0), lat.get("p90_ms", 0))
    log.info("  calib clips used to fit head: %d (never scored above)", int(calib.sum()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["clip_id", "label", "source", "split", "pred", "correct", "confidence"]
    df[cols].to_csv(args.out, index=False)

    metrics.update({
        "model": bb.name, "feature_dim": bb.dim, "n_params": bb.n_params,
        "classifier": kind, "device": device.type,
        "n_calib": int(calib.sum()), "n_eval": int(evalm.sum()),
        "latency_batch1": lat,
    })
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2))

    log.info("\nwrote %s and %s", args.out, args.metrics_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
