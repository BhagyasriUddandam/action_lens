#!/bin/bash
# =============================================================================
# ActionLens — SLURM job template (single GPU node, Northeastern Explorer HPC)
#
# Usage:
#   mkdir -p logs                      # one time; job FAILS instantly if missing
#   sbatch run.slurm data_prep
#   sbatch run.slurm baseline_model
#   sbatch run.slurm vlm_benchmark
#   sbatch run.slurm evaluate
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/actionlens-<jobid>.out
#
# >>> VERIFY BEFORE YOUR FIRST SUBMIT <<<
#   sinfo -s                           # confirm the partition name below exists
#   sinfo -o "%P %G %f"                # confirm GPU types for --constraint
#   module avail anaconda              # confirm the module name/version below
# Cluster module versions change; the three lines flagged VERIFY may need edits.
# =============================================================================

#SBATCH --job-name=actionlens
#SBATCH --partition=gpu                 # VERIFY: `sinfo -s`
#SBATCH --gres=gpu:1                    # single GPU — that is all this project needs
#SBATCH --constraint=a100               # VERIFY: drop this line to accept any GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8               # video decoding is CPU-bound; 8 keeps the GPU fed
#SBATCH --mem=32G
#SBATCH --time=04:00:00                 # raise only if a stage actually needs it
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=uddandam.b@northeastern.edu

set -euo pipefail

# --- argument: which pipeline stage to run --------------------------------
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
    echo "ERROR: no stage given. Usage: sbatch run.slurm <data_prep|baseline_model|vlm_benchmark|evaluate>" >&2
    exit 2
fi

SCRIPT="src/${STAGE}.py"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: $SCRIPT does not exist." >&2
    exit 2
fi

# --- environment -----------------------------------------------------------
module purge
module load anaconda3/2024.06           # VERIFY: `module avail anaconda`

# shellcheck disable=SC1091
source activate actionlens              # env created per README "HPC setup"

# --- provenance: makes every log self-describing ---------------------------
echo "=========================================================="
echo "job id      : ${SLURM_JOB_ID}"
echo "node        : $(hostname)"
echo "stage       : ${STAGE}"
echo "workdir     : $(pwd)"
echo "started     : $(date -Is)"
echo "=========================================================="
nvidia-smi
python -c "import torch; print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')"
echo "----------------------------------------------------------"

# --- run -------------------------------------------------------------------
srun python -u "$SCRIPT" "${@:2}"       # -u: unbuffered, so tail -f shows progress live

echo "----------------------------------------------------------"
echo "finished    : $(date -Is)"
