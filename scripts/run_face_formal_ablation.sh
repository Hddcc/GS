#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ABLATION="${ABLATION:-}"
PHYSICAL_GPU="${PHYSICAL_GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

case "$ABLATION" in
  frequency)
    CONFIG="configs/train/face/train_face_gaussian_frequency_residual_ablation_seed1.yaml"
    NAME="face_gaussian_frequency_residual_ablation_seed1_formal"
    LABEL="frequency-only"
    ;;
  edge)
    CONFIG="configs/train/face/train_face_gaussian_edge_ablation_seed1.yaml"
    NAME="face_gaussian_edge_ablation_seed1_formal"
    LABEL="edge-only"
    ;;
  *)
    echo "Usage: ABLATION=frequency|edge PHYSICAL_GPU=<id> $0" >&2
    exit 2
    ;;
esac

BASELINE_CONFIG="configs/train/face/train_face_gaussian_baseline_seed1.yaml"
FREQUENCY_CONFIG="configs/train/face/train_face_gaussian_frequency_residual_ablation_seed1.yaml"
EDGE_CONFIG="configs/train/face/train_face_gaussian_edge_ablation_seed1.yaml"

python validate_face_formal_ablation_configs.py \
  --baseline "$BASELINE_CONFIG" \
  --frequency "$FREQUENCY_CONFIG" \
  --edge "$EDGE_CONFIG"

echo "START TRAINING: $LABEL"
echo "GPU: $PHYSICAL_GPU"
mkdir -p results
python train_gaussian.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --gpu "$PHYSICAL_GPU" \
  --seed 1 \
  2>&1 | tee "results/${NAME}_train.log"

MODEL="save/${NAME}/epoch-best.pth"
CONFIG_SNAPSHOT="save/${NAME}/config.yaml"
if [[ ! -f "$MODEL" ]]; then
  echo "Missing checkpoint: $MODEL" >&2
  exit 1
fi

OUT_DIR="results/face_formal_ablation/${ABLATION}"
mkdir -p "$OUT_DIR"
SCALES=(1.5 2 2.5 3.5 4 5.5 7.5 8)

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" python test_face_metrics.py \
  --dataset-root Dataset/FACE/CelebA/test/HR \
  --dataset-name CelebA \
  --method model \
  --model "$MODEL" \
  --config "$CONFIG_SNAPSHOT" \
  --scales "${SCALES[@]}" \
  --gpu 0 \
  --num-workers "$NUM_WORKERS" \
  --output "$OUT_DIR/celeba_metrics.csv"

CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU" python test_face_metrics.py \
  --dataset-root Dataset/FACE/Helen/test/HR \
  --dataset-name Helen \
  --method model \
  --model "$MODEL" \
  --config "$CONFIG_SNAPSHOT" \
  --scales "${SCALES[@]}" \
  --gpu 0 \
  --num-workers "$NUM_WORKERS" \
  --output "$OUT_DIR/helen_metrics.csv"

echo "FORMAL ABLATION COMPLETED: $LABEL"
echo "CELEBA=$OUT_DIR/celeba_metrics.csv"
echo "HELEN=$OUT_DIR/helen_metrics.csv"
