#!/usr/bin/env bash
#SBATCH --job-name=h3c_generate
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch h3_causal/scripts/generate.sh [generate.py args], e.g.
#   --weights RUN/exports/step-0001000-ema --data-dir DATA --out OUT --mode causal --num-samples 16 --save-reference

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the MiniMax-H3 repo root}"
ENV_ROOT="${ENV_ROOT:-/mnt/weka/shrd/wm/.conda/wm-production-h200-cu129}"
H3_PYLIBS="${H3_PYLIBS:-/mnt/weka/shrd/wm/mingjia/pylibs/h3}"
export PYTHONPATH="$H3_PYLIBS:$REPO_ROOT:${PYTHONPATH:-}"
export PATH="$ENV_ROOT/bin:$PATH"
export KERNEL_CACHE_ROOT="${KERNEL_CACHE_ROOT:-/tmp/kernel_cache_${USER}_${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="$KERNEL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$KERNEL_CACHE_ROOT/inductor"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
cd "$REPO_ROOT"
exec "$ENV_ROOT/bin/torchrun" --nproc_per_node=8 --master_port="$((20000 + SLURM_JOB_ID % 20000))" -m h3_causal.inference.generate "$@"
