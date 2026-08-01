#!/usr/bin/env bash
# =============================================================================
# fetch_urfd.sh — download the UR Fall Detection dataset (fall sequences only)
#
# Source: http://fenix.ur.edu.pl/~mkepski/ds/uf.html
# License: CC BY-NC-SA 4.0 — non-commercial academic use, attribution required.
# Citation: Kwolek, B., Kepski, M. "Human fall detection on embedded platform
#           using depth maps and wireless accelerometer." Computer Methods and
#           Programs in Biomedicine, 117(3), 2014.
#
# We only need the 30 "fall" sequences (RGB, cam0) — this project does not use
# URFD's ADL sequences (walk/sit/stand come from HMDB51 instead) or depth data.
# data_prep.py later selects 25 of these 30 sequences (seeded, reproducible)
# for the `falling` class; we fetch all 30 here so that choice isn't baked
# into a download script.
#
# Per-frame pose labels (-1 not-lying, 0 falling/transition, 1 lying) live in
# ONE aggregated file covering all 30 sequences — urfall-cam0-falls.csv,
# columns: sequence,frame_num,label,<8 geometry features>, no header.
# Frame numbers in it line up 1:1 with the RGB filenames
# (frame N == fall-NN-cam0-rgb-{N:03d}.png). This is the label source for
# windowing the `falling` class in data_prep.py — NOT the per-sequence
# fall-NN-data.csv (that's a cam/accelerometer sync file, unused here).
#
# Usage:
#   ./scripts/fetch_urfd.sh              # download + extract, keep zips
#   ./scripts/fetch_urfd.sh --clean      # download + extract, delete zips after
#   ./scripts/fetch_urfd.sh --only 01,02 # just these sequence numbers (testing)
#
# Safe to re-run: already-extracted sequences are skipped, partial downloads
# resume with curl -C -.
#
# Total download size: ~1.7 GB across 30 sequences.
# =============================================================================

set -uo pipefail  # no -e: one failed sequence must not abort the other 29

BASE_URL="https://fenix.ur.edu.pl/~mkepski/ds/data"
DEST="data/raw/urfd"
CLEAN=0
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

for cmd in curl unzip; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' is required but not found on PATH." >&2
        exit 1
    fi
done

mkdir -p "$DEST"

if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra SEQ_NUMS <<< "$ONLY"
else
    SEQ_NUMS=($(seq -w 1 30))
fi

echo "=========================================================="
echo "UR Fall Detection dataset — fall sequences (RGB, cam0)"
echo "License: CC BY-NC-SA 4.0 (non-commercial academic use)"
echo "Cite: Kwolek & Kepski, Computer Methods and Programs in"
echo "      Biomedicine, 117(3), 2014."
echo "Fetching ${#SEQ_NUMS[@]} sequence(s) into $DEST/"
echo "=========================================================="

labels_path="${DEST}/urfall-cam0-falls.csv"
if [[ -f "$labels_path" ]]; then
    echo "[labels] urfall-cam0-falls.csv already present — skipping"
else
    echo "[labels] downloading per-frame pose labels (urfall-cam0-falls.csv)..."
    if ! curl -sS -f -C - --retry 3 --retry-delay 5 \
            -o "$labels_path" "${BASE_URL}/urfall-cam0-falls.csv"; then
        echo "[labels] FAILED: could not download urfall-cam0-falls.csv — aborting," \
             "data_prep.py cannot window falls without it" >&2
        exit 1
    fi
fi

ok=0
failed=()

for n in "${SEQ_NUMS[@]}"; do
    seq_id="fall-${n}"
    zip_name="${seq_id}-cam0-rgb.zip"
    extract_dir="${DEST}/${seq_id}-cam0-rgb"
    zip_path="${DEST}/${zip_name}"

    if [[ -d "$extract_dir" ]]; then
        echo "[$seq_id] already present — skipping"
        ok=$((ok + 1))
        continue
    fi

    echo "[$seq_id] downloading RGB zip..."
    if ! curl -sS -f -C - --retry 3 --retry-delay 5 \
            -o "$zip_path" "${BASE_URL}/${zip_name}"; then
        echo "[$seq_id] FAILED: could not download $zip_name" >&2
        failed+=("$seq_id")
        continue
    fi

    echo "[$seq_id] extracting..."
    mkdir -p "$extract_dir"
    if ! unzip -q -o "$zip_path" -d "$extract_dir"; then
        echo "[$seq_id] FAILED: could not extract $zip_name" >&2
        failed+=("$seq_id")
        continue
    fi

    if [[ "$CLEAN" -eq 1 ]]; then
        rm -f "$zip_path"
    fi

    echo "[$seq_id] OK"
    ok=$((ok + 1))
done

echo "=========================================================="
echo "Done: ${ok}/${#SEQ_NUMS[@]} sequences ready in $DEST/"
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "Failed (${#failed[@]}): ${failed[*]}"
    echo "Re-run this script to retry — completed sequences are skipped."
    exit 1
fi
