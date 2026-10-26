#!/usr/bin/env bash
#SBATCH --job-name=h3c_train
#SBATCH --qos=wm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch [--nodes N] h3_causal/scripts/train.sh CONFIG [key=value overrides]
# Resubmitting the same config resumes from the newest complete checkpoint.

set -euo pipefail
CONFIG="$1"
shift
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the MiniMax-H3 repo root}"
ENV_ROOT="${ENV_ROOT:-/mnt/weka/shrd/wm/.conda/wm-production-h200-cu129}"
H3_PYLIBS="${H3_PYLIBS:-/mnt/weka/shrd/wm/mingjia/pylibs/h3}"
NODES="${SLURM_NNODES:?submit through sbatch}"
test -f "$CONFIG"
export REPO_ROOT ENV_ROOT NODES
export PYTHONPATH="$H3_PYLIBS:$REPO_ROOT:${PYTHONPATH:-}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
# Node-local compiler caches (a shared Triton/Inductor cache on Weka races into ESTALE).
export KERNEL_CACHE_ROOT="${KERNEL_CACHE_ROOT:-/tmp/kernel_cache_${USER}_${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="$KERNEL_CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$KERNEL_CACHE_ROOT/inductor"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$REPO_ROOT"
srun --kill-on-bad-exit=1 --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 \
    bash -c 'mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" &&
    exec "$ENV_ROOT/bin/torchrun" --nnodes="$NODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
        -m h3_causal.train.trainer "$@"' bash --config "$CONFIG" "$@"
