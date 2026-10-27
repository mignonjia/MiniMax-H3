#!/usr/bin/env bash
#SBATCH --job-name=h3c_caption
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Usage: sbatch [--nodes N] h3_causal/scripts/caption.sh MANIFEST CAPTION_DIR MERGED_MANIFEST [chunk_captioner.py args]
# Restartable: ranks skip ids already in CAPTION_DIR/captions.rank*.jsonl; the merged manifest is written last.

set -euo pipefail
MANIFEST="$1"
OUT="$2"
MERGED="$3"
shift 3
REPO_ROOT="${SLURM_SUBMIT_DIR:?submit from the MiniMax-H3 repo root}"
ENV_ROOT="${ENV_ROOT:-/mnt/weka/shrd/wm/.conda/wm-production-h200-cu129}"
H3_PYLIBS="${H3_PYLIBS:-/mnt/weka/shrd/wm/mingjia/pylibs/h3}"
NODES="${SLURM_NNODES:?submit through sbatch}"
export REPO_ROOT ENV_ROOT NODES
export PYTHONPATH="$H3_PYLIBS:$REPO_ROOT:${PYTHONPATH:-}"
export PATH="$ENV_ROOT/bin:$PATH"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
cd "$REPO_ROOT"
srun --nodes="$NODES" --ntasks="$NODES" --ntasks-per-node=1 bash -c 'exec "$ENV_ROOT/bin/torchrun" \
    --nnodes="$NODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -m h3_causal.preprocess.chunk_captioner "$@"' bash --manifest "$MANIFEST" --out "$OUT" "$@"
"$ENV_ROOT/bin/python" -m h3_causal.preprocess.chunk_captioner --out "$OUT" --merge "$MERGED"
