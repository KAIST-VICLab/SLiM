#!/usr/bin/env bash
# SLiM self-supervised pre-training.
#
#   scripts/pretrain.sh <run> <output_dir> [gpu_ids] [extra key=value ...]
#
# <run> is one of:  ntu60_xsub  ntu60_xview  ntu120_xsub  ntu120_xset
#
# One pre-training per evaluation protocol, on that protocol's training clips only,
# so the protocol's test clips are never seen before evaluation.
#
# Effective batch size is 768 = 4 GPUs x 192 clips. Change train.batch_size_per_gpu
# if you use a different number of GPUs, or the LR scaling rule will not match.
# An epoch is a fixed 1250 iterations (train.OFFICIAL_EPOCH_LENGTH), so the schedule
# and total compute are the same for every run.
# Training resumes automatically from the last checkpoint in <output_dir>.
set -euo pipefail

DATASET=${1:?usage: pretrain.sh <ntu60_xsub|ntu60_xview|ntu120_xsub|ntu120_xset> <output_dir> [gpu_ids] [opts...]}
OUTDIR=${2:?missing output_dir}
GPUS=${3:-0,1,2,3}
shift 3 || shift 2
EXTRA=("$@")

DATA_ROOT=${DATA_ROOT:-./data}
case "$DATASET" in
  ntu60_xsub)   NPZ="$DATA_ROOT/ntu/NTU60_XSub.npz" ;;
  ntu60_xview)  NPZ="$DATA_ROOT/ntu/NTU60_XView.npz" ;;
  ntu120_xsub)  NPZ="$DATA_ROOT/ntu120/NTU120_CS.npz" ;;
  ntu120_xset)  NPZ="$DATA_ROOT/ntu120/NTU120_CV.npz" ;;
  *) echo "unknown run: $DATASET (expected ntu60_xsub, ntu60_xview, ntu120_xsub or ntu120_xset)" >&2; exit 1 ;;
esac

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 KMP_AFFINITY=none MKL_THREADING_LAYER=GNU \
  torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29500}" \
  -m slim.train.train \
  --config-file "slim/configs/pretrain/slim_${DATASET}.yaml" \
  --output-dir "$OUTDIR" \
  "train.dataset_path=NTU:root=${NPZ}:split=TRAIN" \
  ${EXTRA[@]+"${EXTRA[@]}"}
