#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PHYSICAL_GPUS="${PHYSICAL_GPUS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IFS=',' read -r GPU_FREQUENCY GPU_EDGE EXTRA_GPU <<< "$PHYSICAL_GPUS"
if [[ -n "${EXTRA_GPU:-}" || -z "${GPU_FREQUENCY:-}" || -z "${GPU_EDGE:-}" ]]; then
  echo "PHYSICAL_GPUS must contain exactly two GPU ids, for example 3,4." >&2
  exit 2
fi

mkdir -p results/face_formal_ablation
echo "FORMAL ABLATION PAIR STARTED"
echo "frequency-only GPU: $GPU_FREQUENCY"
echo "edge-only GPU: $GPU_EDGE"

ABLATION=frequency PHYSICAL_GPU="$GPU_FREQUENCY" NUM_WORKERS="$NUM_WORKERS" \
  bash scripts/run_face_formal_ablation.sh \
  > results/face_formal_ablation/frequency_orchestrator.log 2>&1 &
PID_FREQUENCY=$!

ABLATION=edge PHYSICAL_GPU="$GPU_EDGE" NUM_WORKERS="$NUM_WORKERS" \
  bash scripts/run_face_formal_ablation.sh \
  > results/face_formal_ablation/edge_orchestrator.log 2>&1 &
PID_EDGE=$!

echo "frequency-only pid: $PID_FREQUENCY"
echo "edge-only pid: $PID_EDGE"
set +e
wait "$PID_FREQUENCY"
STATUS_FREQUENCY=$?
wait "$PID_EDGE"
STATUS_EDGE=$?
set -e

if [[ "$STATUS_FREQUENCY" -ne 0 || "$STATUS_EDGE" -ne 0 ]]; then
  echo "FORMAL ABLATION PAIR FAILED: frequency=$STATUS_FREQUENCY edge=$STATUS_EDGE" >&2
  exit 1
fi

echo "FORMAL ABLATION PAIR COMPLETED"
