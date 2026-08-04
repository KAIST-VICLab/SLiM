#!/usr/bin/env bash
# Action retrieval (Table 2b): non-parametric k-NN on frozen features (k = 1).
#
#   scripts/eval_knn.sh <checkpoint> <output_dir> <ntu60|ntu120> <xsub|xview> [gpu_ids]
set -euo pipefail

CKPT=${1:?usage: eval_knn.sh <checkpoint> <output_dir> <ntu60|ntu120> <xsub|xview> [gpu_ids]}
OUTDIR=${2:?missing output_dir}
DATASET=${3:?missing dataset}
BENCH=${4:?missing benchmark}
GPUS=${5:-0,1,2,3}

DATA_ROOT=${DATA_ROOT:-./data}
case "$DATASET" in
  ntu60)  DATA_PATH="$DATA_ROOT/ntu";    NUM_CLASSES=60 ;;
  ntu120) DATA_PATH="$DATA_ROOT/ntu120"; NUM_CLASSES=120 ;;
  *) echo "unknown dataset: $DATASET (expected ntu60 or ntu120)" >&2; exit 1 ;;
esac

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 KMP_AFFINITY=none MKL_THREADING_LAYER=GNU \
  torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29503}" \
  -m slim.eval.knn \
  --config-file slim/configs/eval/slim_eval.yaml \
  --pretrained-weights "$CKPT" \
  --output-dir "$OUTDIR" \
  --data-path "$DATA_PATH" \
  --num-classes "$NUM_CLASSES" \
  --benchmark "$BENCH"
