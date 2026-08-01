#!/usr/bin/env bash
# =============================================================================
# fetch_hmdb51.sh — download the 4 HMDB51 classes ActionLens needs
#
# Classes: walk, sit, stand, fall_floor  ->  data/raw/hmdb51/{class}/*.avi
#
# WHY NOT THE OFFICIAL BROWN URL:
#   https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
#   is DEAD (verified 2026-07-31). Brown migrated the lab site off WordPress to
#   GitHub Pages: the old host 301-redirects to the new homepage and the
#   wp-content path 404s on the new domain. This is NOT a user-agent block —
#   browser UA / referer / -L all fail identically because the file is gone.
#   The Google Drive mirror linked from the current lab page also 404s.
#
# SOURCE USED (default):
#   HF dataset CVML-TueAI/HMDB51 — a complete mirror (6766 clips, all 51
#   classes) that stores INDIVIDUAL .avi files under hmdb51_org/{class}/,
#   not archives. Ungated, no auth. This means we fetch only our 4 classes
#   (~292 MB / ~980 clips) instead of the full 2.1 GB rar, and need no unrar.
#
# FALLBACK SOURCE (--method rar):
#   HF dataset Serrelab/hmdb51 — Brown's own HF account, the authoritative
#   hmdb51_org.rar (2.1 GB, rar-of-rars). Requires `unrar`, which is NOT
#   installed by default on macOS or Explorer. Kept as insurance only.
#
# We download ALL clips for these 4 classes, not just the 40/class the project
# uses. The seeded 40-per-class selection belongs in data_prep.py, so it is not
# baked into a download script.
#
# Usage:
#   ./scripts/fetch_hmdb51.sh                     # all 4 classes (recommended)
#   ./scripts/fetch_hmdb51.sh --only walk,sit     # subset, for testing
#   ./scripts/fetch_hmdb51.sh --method rar        # official rar fallback
#
# Safe to re-run: existing files are skipped.
# =============================================================================

set -uo pipefail  # no -e: one failed class/file must not abort the rest

HF_REPO="CVML-TueAI/HMDB51"
HF_RAR_REPO="Serrelab/hmdb51"
DEST="data/raw/hmdb51"
CLASSES=(walk sit stand fall_floor)
METHOD="files"
ONLY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)   ONLY="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "$ONLY" ]]; then
    IFS=',' read -ra CLASSES <<< "$ONLY"
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: 'curl' is required but not found on PATH." >&2
    exit 1
fi

mkdir -p "$DEST"

# -----------------------------------------------------------------------------
# fallback path: official rar-of-rars from Brown's own HF account
# -----------------------------------------------------------------------------
fetch_via_rar() {
    echo "Method: official rar (${HF_RAR_REPO})"

    local unrar_bin=""
    for c in unrar 7z 7za; do
        if command -v "$c" >/dev/null 2>&1; then unrar_bin="$c"; break; fi
    done

    if [[ -z "$unrar_bin" ]]; then
        cat >&2 <<'EOF'
ERROR: this method needs `unrar` (or 7z) and none is installed.

Install one of:
    conda install -c conda-forge unrar     # works on macOS and Explorer
    brew install carlocab/personal/unrar   # macOS only

Or just use the default method, which needs no archive tool at all:
    ./scripts/fetch_hmdb51.sh
EOF
        return 1
    fi

    local outer="${DEST}/hmdb51_org.rar"
    if [[ ! -f "$outer" ]]; then
        echo "Downloading hmdb51_org.rar (2.1 GB)..."
        if ! curl -sS -f -L -C - --retry 3 --retry-delay 5 --progress-bar \
                -o "$outer" \
                "https://huggingface.co/datasets/${HF_RAR_REPO}/resolve/main/hmdb51_org.rar"; then
            echo "ERROR: failed to download hmdb51_org.rar" >&2
            return 1
        fi
    else
        echo "hmdb51_org.rar already present — skipping download"
    fi

    # outer rar contains one <class>.rar per class; extract only ours
    for cls in "${CLASSES[@]}"; do
        echo "[$cls] extracting ${cls}.rar from outer archive..."
        "$unrar_bin" e -o+ -inul "$outer" "${cls}.rar" "$DEST/" 2>/dev/null \
            || echo "[$cls] WARN: could not extract ${cls}.rar" >&2
        if [[ -f "${DEST}/${cls}.rar" ]]; then
            mkdir -p "${DEST}/${cls}"
            "$unrar_bin" e -o+ -inul "${DEST}/${cls}.rar" "${DEST}/${cls}/" \
                || echo "[$cls] WARN: could not extract clips" >&2
            rm -f "${DEST}/${cls}.rar"
            echo "[$cls] $(find "${DEST}/${cls}" -name '*.avi' | wc -l | tr -d ' ') clips"
        fi
    done
}

# -----------------------------------------------------------------------------
# default path: per-file download of only the classes we need
# -----------------------------------------------------------------------------
fetch_via_files() {
    echo "Method: per-file (${HF_REPO}) — no archive tool needed"

    # Preferred: huggingface_hub handles resume, retries and caching for us.
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Using huggingface_hub (resumable, cached)"
        CLASSES_CSV="$(IFS=,; echo "${CLASSES[*]}")" DEST="$DEST" HF_REPO="$HF_REPO" \
        python3 <<'PY'
import os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

classes = os.environ["CLASSES_CSV"].split(",")
dest = Path(os.environ["DEST"])
repo = os.environ["HF_REPO"]

patterns = [f"hmdb51_org/{c}/*.avi" for c in classes]
print(f"Downloading {len(classes)} class(es): {', '.join(classes)}")

try:
    snap = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=patterns,
        max_workers=8,
    )
except Exception as e:
    print(f"ERROR: snapshot_download failed: {e}", file=sys.stderr)
    sys.exit(1)

# Materialise into data/raw/hmdb51/{class}/ — the cache uses symlinks, and
# downstream code (and HPC scratch copies) should not depend on those.
total = 0
for c in classes:
    src = Path(snap) / "hmdb51_org" / c
    if not src.is_dir():
        print(f"[{c}] WARN: not found in mirror", file=sys.stderr)
        continue
    out = dest / c
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for avi in sorted(src.glob("*.avi")):
        target = out / avi.name
        if not target.exists():
            shutil.copyfile(avi, target)  # resolves symlink -> real bytes
        n += 1
    total += n
    print(f"[{c}] {n} clips")
print(f"TOTAL: {total} clips")
PY
        return $?
    fi

    # Fallback: no huggingface_hub (e.g. a bare env on Explorer) -> curl loop.
    echo "huggingface_hub not available — falling back to curl"
    local rc=0
    for cls in "${CLASSES[@]}"; do
        echo "[$cls] listing files..."
        local listing
        listing="$(curl -sS -f --max-time 60 \
            "https://huggingface.co/api/datasets/${HF_REPO}/tree/main/hmdb51_org/${cls}?recursive=1" \
            | python3 -c "
import json,sys
try:
    for x in json.load(sys.stdin):
        p = x.get('path','')
        if p.endswith('.avi'):
            print(p)
except Exception as e:
    sys.exit(1)
")"
        if [[ -z "$listing" ]]; then
            echo "[$cls] FAILED: could not list files" >&2
            rc=1
            continue
        fi

        mkdir -p "${DEST}/${cls}"
        local n=0 got=0
        while IFS= read -r path; do
            [[ -z "$path" ]] && continue
            n=$((n + 1))
            local fname="${path##*/}"
            local out="${DEST}/${cls}/${fname}"
            [[ -s "$out" ]] && { got=$((got + 1)); continue; }
            if curl -sS -f -L -C - --retry 3 --retry-delay 2 \
                    -o "$out" \
                    "https://huggingface.co/datasets/${HF_REPO}/resolve/main/${path}"; then
                got=$((got + 1))
            else
                echo "[$cls] WARN: failed $fname" >&2
                rm -f "$out"
            fi
            if (( n % 50 == 0 )); then echo "[$cls] $got/$n..."; fi
        done <<< "$listing"
        echo "[$cls] $got/$n clips"
        (( got == 0 )) && rc=1
    done
    return $rc
}

echo "=========================================================="
echo "HMDB51 — ActionLens subset"
echo "Classes: ${CLASSES[*]}"
echo "License: CC BY 4.0 (attribution required)"
echo "Cite: Kuehne et al., 'HMDB: A Large Video Database for"
echo "      Human Motion Recognition', ICCV 2011."
echo "Destination: ${DEST}/"
echo "=========================================================="

case "$METHOD" in
    files) fetch_via_files ;;
    rar)   fetch_via_rar ;;
    *) echo "Unknown --method '$METHOD' (expected 'files' or 'rar')" >&2; exit 2 ;;
esac
status=$?

echo "=========================================================="
if [[ $status -ne 0 ]]; then
    echo "Completed WITH ERRORS — re-run to retry (existing files are skipped)."
    exit 1
fi
for cls in "${CLASSES[@]}"; do
    if [[ -d "${DEST}/${cls}" ]]; then
        printf "  %-12s %s clips\n" "$cls" \
            "$(find "${DEST}/${cls}" -name '*.avi' | wc -l | tr -d ' ')"
    fi
done
echo "Done."
