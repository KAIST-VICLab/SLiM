#!/usr/bin/env bash
# Semi-supervised evaluation (Table 2a): end-to-end fine-tuning on a
# class-balanced fraction of the labelled training set.
#
#   scripts/eval_semi.sh <checkpoint> <output_dir> <ntu60|ntu120> <xsub|xview> <ratio> [gpu_ids]
#
# ratio 0.01 = 1 % of the labels, 0.1 = 10 %.
#
# Learning rates are scaled by batch_size * world_size / 256, so the reported
# setting is batch 128 on 2 GPUs (effective head LR 1e-3, backbone LR 1e-5).
# Changing the GPU count changes the effective LR.
set -euo pipefail

CKPT=${1:?usage: eval_semi.sh <checkpoint> <output_dir> <ntu60|ntu120> <xsub|xview> <ratio> [gpu_ids]}
OUTDIR=${2:?missing output_dir}
DATASET=${3:?missing dataset}
BENCH=${4:?missing benchmark}
RATIO=${5:?missing ratio}
GPUS=${6:-0,1}

DATA_ROOT=${DATA_ROOT:-./data}
case "$DATASET" in
  ntu60)  DATA_PATH="$DATA_ROOT/ntu";    NUM_CLASSES=60 ;;
  ntu120) DATA_PATH="$DATA_ROOT/ntu120"; NUM_CLASSES=120 ;;
  *) echo "unknown dataset: $DATASET (expected ntu60 or ntu120)" >&2; exit 1 ;;
esac

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 KMP_AFFINITY=none MKL_THREADING_LAYER=GNU \
  torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29502}" \
  -m slim.eval.finetune \
  --config-file slim/configs/eval/slim_eval.yaml \
  --pretrained-weights "$CKPT" \
  --output-dir "$OUTDIR" \
  --data-path "$DATA_PATH" \
  --num-classes "$NUM_CLASSES" \
  --benchmark "$BENCH" \
  --sample-ratio "$RATIO" \
  --head-lr 1e-3 \
  --backbone-lr 1e-5
