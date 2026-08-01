#!/usr/bin/env python3
"""Baseline vs VLM: the head-to-head comparison and the dashboard's data source.

Reads results/baseline_predictions.csv + results/vlm_predictions.csv (produced
by baseline_model.py and vlm_benchmark.py), joins them on clip_id, and writes:

    results/metrics.csv         -- wide comparison table (class x approach x metric)
    results/failure_cases.csv   -- every disagreement, with VLM evidence text
    results/plots/*.png         -- per-class accuracy, confusion matrices, cost/latency
    results/results.json        -- the dashboard reads this; nothing else

Both prediction files carry a `split` column; this script filters to
split == "eval" for every reported number regardless of how each file was
generated, so a stray calib-split row can never leak into the comparison.

Every headline number here (the biggest per-class gap, whether VLM separates
sit/stand) is COMPUTED from the joined data, not hardcoded -- rerunning with a
different prompt or dataset must change the dashboard's story, not just its
numbers.

Run:
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

log = logging.getLogger("evaluate")

LABELS = ("walking", "sitting", "standing", "falling")

# Categorical pair from the project's validated palette (dataviz skill):
# slot 1 blue / slot 2 orange, the CVD-safe adjacent pair. Reused unchanged in
# the dashboard so tables, plots, and charts read as one system.
COLOR_BASELINE = "#2a78d6"
COLOR_VLM = "#eb6834"
# Sequential ramp (single hue, light->dark) for confusion-matrix magnitude.
SEQUENTIAL_BLUE = ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def load_predictions(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{name} predictions not found: {path}\n"
            f"Run the {name} script first (see tasks/todo.md Checkpoint B)."
        )
    df = pd.read_csv(path)
    df = df[df.split == "eval"].copy()
    if df.empty:
        raise ValueError(f"{path} has no split=='eval' rows")
    return df


def per_class_metrics(df: pd.DataFrame, labels: tuple[str, ...]) -> dict:
    scored = df[df.pred.notna()]
    y_true, y_pred = scored.label.to_numpy(), scored.pred.to_numpy()
    rep = classification_report(y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0)
    out = {}
    for c in labels:
        sub = scored[scored.label == c]
        acc = float((sub.label == sub.pred).mean()) if len(sub) else float("nan")
        out[c] = {
            "accuracy": round(acc, 4),
            "precision": round(rep[c]["precision"], 4),
            "recall": round(rep[c]["recall"], 4),
            "f1": round(rep[c]["f1-score"], 4),
            "n": int(rep[c]["support"]),
        }
    return out


def overall_accuracy(df: pd.DataFrame) -> float:
    scored = df[df.pred.notna()]
    return float((scored.label == scored.pred).mean()) if len(scored) else float("nan")


def confusion(df: pd.DataFrame, labels: tuple[str, ...]) -> list[list[int]]:
    scored = df[df.pred.notna()]
    return confusion_matrix(scored.label, scored.pred, labels=list(labels)).tolist()


def falling_by_source(df: pd.DataFrame) -> dict:
    scored = df[(df.pred.notna()) & (df.label == "falling")]
    out = {}
    for src, g in scored.groupby("source"):
        out[src] = {"accuracy": round(float((g.label == g.pred).mean()), 4), "n": len(g)}
    return out


# --- comparison ----------------------------------------------------------------


def build_comparison(
    base: pd.DataFrame, vlm: pd.DataFrame, labels: tuple[str, ...], failures: pd.DataFrame
) -> dict:
    base_pc = per_class_metrics(base, labels)
    vlm_pc = per_class_metrics(vlm, labels)

    gap = {c: round(vlm_pc[c]["accuracy"] - base_pc[c]["accuracy"], 4) for c in labels}
    largest_gap_class = max(gap, key=lambda c: gap[c])

    # The sit/stand question this whole benchmark is built to answer: does the
    # VLM separate the two classes the frozen baseline collapsed? "Separates"
    # here means both classes individually clear a coin-flip (0.5) -- matching
    # accuracy alone would hide one class dragging the other up.
    sit_stand_gap = {
        "baseline_sitting_acc": base_pc["sitting"]["accuracy"],
        "baseline_standing_acc": base_pc["standing"]["accuracy"],
        "vlm_sitting_acc": vlm_pc["sitting"]["accuracy"],
        "vlm_standing_acc": vlm_pc["standing"]["accuracy"],
        "vlm_separates_sit_stand": bool(
            vlm_pc["sitting"]["accuracy"] > 0.5 and vlm_pc["standing"]["accuracy"] > 0.5
        ),
    }

    # Full bucket counts, not just the curated examples in failure_cases -- the
    # dashboard's gallery headings ("VLM corrected N of M mistakes") need the
    # real totals, computed here rather than read off a log line by a human.
    bucket_counts = failures.bucket.value_counts().to_dict()

    return {
        "per_class_accuracy_gap": gap,
        "largest_gap_class": largest_gap_class,
        "sit_stand": sit_stand_gap,
        "failure_bucket_counts": {
            "vlm_correct_baseline_wrong": int(bucket_counts.get("vlm_correct_baseline_wrong", 0)),
            "baseline_correct_vlm_wrong": int(bucket_counts.get("baseline_correct_vlm_wrong", 0)),
            "both_wrong": int(bucket_counts.get("both_wrong", 0)),
        },
        "n_failures_total": int(len(failures)),
    }


def build_failure_cases(base: pd.DataFrame, vlm: pd.DataFrame) -> pd.DataFrame:
    """Every eval clip where at least one approach got it wrong, for FINDINGS.md.

    Outer-joined on clip_id: a clip missing from one side (e.g. a VLM refusal)
    still appears, with that side's prediction blank rather than silently
    dropped.
    """
    b = base[["clip_id", "label", "source", "pred", "confidence"]].rename(
        columns={"pred": "baseline_pred", "confidence": "baseline_confidence"}
    )
    v = vlm[["clip_id", "pred", "confidence", "evidence"]].rename(
        columns={"pred": "vlm_pred", "confidence": "vlm_confidence"}
    )
    m = b.merge(v, on="clip_id", how="outer")

    baseline_wrong = m.baseline_pred != m.label
    vlm_wrong = m.vlm_pred != m.label
    m = m[baseline_wrong | vlm_wrong].copy()

    def bucket(row):
        if row.baseline_pred != row.label and row.vlm_pred != row.label:
            return "both_wrong"
        if row.baseline_pred != row.label:
            return "vlm_correct_baseline_wrong"
        return "baseline_correct_vlm_wrong"

    m["bucket"] = m.apply(bucket, axis=1)
    return m.sort_values(["bucket", "label", "clip_id"]).reset_index(drop=True)


# --- plots -----------------------------------------------------------------
# Matplotlib static PNGs for FINDINGS.md / README -- separate from the
# interactive dashboard, which renders its own charts from results.json.
# Palette, form choices, and "one axis, never two" follow the dataviz skill.


def plot_per_class_accuracy(base_pc: dict, vlm_pc: dict, labels: tuple[str, ...], out: Path) -> None:
    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = [base_pc[c]["accuracy"] * 100 for c in labels]
    v = [vlm_pc[c]["accuracy"] * 100 for c in labels]
    bars_b = ax.bar(x - width / 2, b, width, label="Baseline", color=COLOR_BASELINE)
    bars_v = ax.bar(x + width / 2, v, width, label="VLM", color=COLOR_VLM)
    for bars in (bars_b, bars_v):
        ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=9)
    ax.set_xticks(x, [c.capitalize() for c in labels])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(lambda v, _: f"{int(v)}%")
    ax.set_title("Per-class accuracy: baseline vs VLM")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(base_cm, vlm_cm, labels: tuple[str, ...], out: Path) -> None:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cm, title in zip(axes, (base_cm, vlm_cm), ("Baseline", "VLM")):
        cm = np.array(cm)
        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks(range(len(labels)), [c[:4].capitalize() for c in labels], rotation=0)
        ax.set_yticks(range(len(labels)), [c[:4].capitalize() for c in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        vmax = cm.max() if cm.max() > 0 else 1
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Dark text on light cells, light text on dark cells.
                color = "white" if cm[i, j] > vmax * 0.6 else "#0b0b0b"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=10)
    fig.suptitle("Confusion matrices (rows = true label)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_latency(base_latency_ms: float, vlm_latency_ms: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Baseline", "VLM"], [base_latency_ms, vlm_latency_ms],
                   color=[COLOR_BASELINE, COLOR_VLM])
    ax.bar_label(bars, fmt="%.0f ms", padding=3)
    ax.set_ylabel("Latency per clip (ms)")
    ax.set_title("Inference latency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cost(vlm_cost_per_1k: float, out: Path) -> None:
    # Single series: baseline runs on owned GPU compute, not a metered API, so
    # there is no honest per-clip dollar figure to place beside it -- see the
    # note on baseline cost in results.json instead of faking a $0 bar here.
    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(["VLM"], [vlm_cost_per_1k], color=COLOR_VLM, width=0.5)
    ax.bar_label(bars, fmt="$%.2f", padding=3)
    ax.set_ylabel("Cost per 1,000 clips (USD)")
    ax.set_title("VLM API cost\n(baseline runs on owned GPU compute)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --- results.json ------------------------------------------------------------


_MID_SENTENCE_GLUE = re.compile(r"\.[a-z]")  # period immediately followed by
# lowercase, no space -- a sentence boundary with no gap, the signature of
# appended trailing text rather than a deliberate abbreviation.


def has_clean_evidence(text) -> bool:
    """True unless the VLM's evidence field looks truncated or malformed.

    Rare (2/128 eval predictions in practice, verified by scanning the full
    set): the model occasionally runs on past its evidence field near the
    length boundary and appends a stray fragment -- brace/bracket characters,
    or a second clause glued on with no space after the period. Confirmed
    this never touches `pred`/`confidence` (checked separately), so it does
    not affect any reported accuracy number -- only curation (which few of
    many examples to show) is affected here.
    """
    if not isinstance(text, str) or not text:
        return False
    if any(c in text for c in "{}[]"):
        return False
    return not _MID_SENTENCE_GLUE.search(text)


def build_results_json(
    base: pd.DataFrame, vlm: pd.DataFrame, base_metrics: dict, vlm_metrics: dict,
    labels: tuple[str, ...], comparison: dict, failures: pd.DataFrame,
) -> dict:
    n_eval = len(base)
    falling = base[base.label == "falling"]
    falling_sources = falling.source.value_counts().to_dict()

    # A small curated set for the dashboard, not all ~40 failures -- full list
    # is in failure_cases.csv for FINDINGS.md.
    curated = []
    for bucket in ("vlm_correct_baseline_wrong", "both_wrong", "baseline_correct_vlm_wrong"):
        pool = failures[failures.bucket == bucket]
        clean = pool[pool.evidence.apply(has_clean_evidence)]
        # Prefer clean examples; only fall back to a malformed one if a
        # bucket has fewer than 3 clean candidates.
        sub = pd.concat([clean, pool[~pool.index.isin(clean.index)]]).head(3)
        for r in sub.itertuples():
            curated.append({
                "clip_id": r.clip_id, "label": r.label, "source": r.source,
                "baseline_pred": r.baseline_pred, "vlm_pred": r.vlm_pred,
                "vlm_evidence": r.evidence if isinstance(r.evidence, str) else "",
                "bucket": bucket,
            })

    return {
        "dataset": {
            "classes": list(labels),
            "n_eval": n_eval,
            "n_calib": 160 - n_eval if n_eval < 160 else None,
            "falling_sources": falling_sources,
        },
        "approaches": {
            "baseline": {
                "name": "Pretrained action-recognition backbone (frozen) + linear probe",
                "model": base_metrics.get("model"),
                "overall_accuracy": round(overall_accuracy(base), 4),
                "per_class": per_class_metrics(base, labels),
                "confusion_matrix": {"labels": list(labels), "matrix": confusion(base, labels)},
                "falling_by_source": falling_by_source(base),
                "latency_ms_per_clip": base_metrics.get("latency_batch1", {}).get("mean_ms"),
                "cost_note": "Runs on owned GPU compute -- no per-call API cost to report.",
            },
            "vlm": {
                "name": f"{vlm_metrics.get('model', 'VLM')} (vision-language model)",
                "model": vlm_metrics.get("model"),
                "overall_accuracy": round(overall_accuracy(vlm), 4),
                "per_class": per_class_metrics(vlm, labels),
                "confusion_matrix": {"labels": list(labels), "matrix": confusion(vlm, labels)},
                "falling_by_source": falling_by_source(vlm),
                "latency_ms_per_clip": vlm_metrics.get("cost", {}).get("latency_mean_ms"),
                "cost_per_1k_clips_usd": vlm_metrics.get("cost", {}).get("per_1k_clips_usd"),
            },
        },
        "comparison": comparison,
        "failure_cases": curated,
    }


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-csv", type=Path, default=Path("results/baseline_predictions.csv"))
    p.add_argument("--vlm-csv", type=Path, default=Path("results/vlm_predictions.csv"))
    p.add_argument("--baseline-metrics", type=Path, default=Path("results/baseline_metrics.json"))
    p.add_argument("--vlm-metrics", type=Path, default=Path("results/vlm_metrics.json"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    try:
        base = load_predictions(args.baseline_csv, "baseline_model.py")
        vlm = load_predictions(args.vlm_csv, "vlm_benchmark.py")
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        return 1

    base_metrics = json.loads(args.baseline_metrics.read_text()) if args.baseline_metrics.is_file() else {}
    vlm_metrics = json.loads(args.vlm_metrics.read_text()) if args.vlm_metrics.is_file() else {}

    common = sorted(set(base.clip_id) & set(vlm.clip_id))
    if len(common) != len(base) or len(common) != len(vlm):
        log.warning(
            "clip_id mismatch: baseline has %d eval rows, vlm has %d, %d in common "
            "-- comparison uses the intersection", len(base), len(vlm), len(common),
        )
    base = base[base.clip_id.isin(common)].reset_index(drop=True)
    vlm = vlm[vlm.clip_id.isin(common)].reset_index(drop=True)

    labels = LABELS
    base_pc = per_class_metrics(base, labels)
    vlm_pc = per_class_metrics(vlm, labels)
    failures = build_failure_cases(base, vlm)
    comparison = build_comparison(base, vlm, labels, failures)

    log.info("=" * 62)
    log.info("BASELINE vs VLM  (n=%d eval clips)", len(common))
    log.info("=" * 62)
    log.info("  %-10s %10s %10s %8s", "class", "baseline", "vlm", "gap")
    for c in labels:
        log.info("  %-10s %9.1f%% %9.1f%% %+7.1f%%", c, base_pc[c]["accuracy"] * 100,
                 vlm_pc[c]["accuracy"] * 100, comparison["per_class_accuracy_gap"][c] * 100)
    log.info("  %-10s %9.1f%% %9.1f%%", "OVERALL", overall_accuracy(base) * 100, overall_accuracy(vlm) * 100)
    log.info("\n  largest gap: %s (VLM %+.1f%% vs baseline)",
             comparison["largest_gap_class"], comparison["per_class_accuracy_gap"][comparison["largest_gap_class"]] * 100)
    log.info("  VLM separates sitting/standing: %s", comparison["sit_stand"]["vlm_separates_sit_stand"])
    log.info("  failure cases: %d (%s)", len(failures), failures.bucket.value_counts().to_dict())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # metrics.csv -- wide comparison table
    rows = []
    for c in labels:
        for approach, pc in (("baseline", base_pc), ("vlm", vlm_pc)):
            rows.append({"class": c, "approach": approach, **pc[c]})
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False)

    # failure_cases.csv -- full list for FINDINGS.md
    failures.to_csv(args.out_dir / "failure_cases.csv", index=False)

    # plots
    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_per_class_accuracy(base_pc, vlm_pc, labels, plots_dir / "per_class_accuracy.png")
    plot_confusion_matrices(confusion(base, labels), confusion(vlm, labels), labels,
                             plots_dir / "confusion_matrices.png")
    base_lat = base_metrics.get("latency_batch1", {}).get("mean_ms", 0.0)
    vlm_lat = vlm_metrics.get("cost", {}).get("latency_mean_ms", 0.0)
    plot_latency(base_lat, vlm_lat, plots_dir / "latency.png")
    vlm_cost = vlm_metrics.get("cost", {}).get("per_1k_clips_usd", 0.0)
    plot_cost(vlm_cost, plots_dir / "cost.png")

    # results.json -- the dashboard's ONLY data source
    results = build_results_json(base, vlm, base_metrics, vlm_metrics, labels, comparison, failures)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))

    log.info("\nwrote metrics.csv, failure_cases.csv, results.json, and 4 plots to %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
