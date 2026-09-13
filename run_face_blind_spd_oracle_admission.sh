#!/usr/bin/env bash
set -euo pipefail

cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main

physical_gpu="${PHYSICAL_GPU:-4}"
eval_bsize="${EVAL_BSIZE:-20000}"
baseline=save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth
config=configs/train/face/probe_face_gaussian_baseline_seed1.yaml
protocol=configs/eval/face_blind_spd_oracle_protocol_20260913.json
helen_root=Dataset/FACE/Helen/test/HR
output=results/face_blind_spd_oracle_admission
result="$output/oracle_result.json"

[[ "$physical_gpu" =~ ^(0|2|4)$ ]] || {
  echo 'STOP: PHYSICAL_GPU must be one of the available cards: 0, 2, 4' >&2
  exit 1
}
[[ "$eval_bsize" =~ ^[1-9][0-9]*$ ]] || {
  echo 'STOP: EVAL_BSIZE must be a positive integer' >&2
  exit 1
}
previous_log=results/face_multi_primitive_m4_seed1_probe_orchestrator.log
[[ -f "$previous_log" ]] && grep -qx \
  'MULTI-PRIMITIVE SEED1 PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: multi-primitive probe has not reached a terminal state' >&2
  exit 1
}
for required in "$baseline" "$config" "$protocol"; do
  [[ -f "$required" ]] || {
    echo "STOP: missing required file: $required" >&2
    exit 1
  }
done
for directory in \
    Dataset/FACE/CelebA/val/HR \
    "$helen_root"; do
  [[ -d "$directory" ]] || {
    echo "STOP: missing dataset directory: $directory" >&2
    exit 1
  }
done
[[ ! -e "$output" ]] || {
  echo "STOP: output already exists; do not overwrite: $output" >&2
  exit 1
}

mkdir -p "$output"
echo "RUNTIME: physical GPU $physical_gpu, evaluation only, no training"
echo 'PROTOCOL: 18-image sweep; conditional 100 CelebA + 50 Helen confirmation'
echo 'VARIANTS: HR/LR mapping x add/subtract x alpha 0.25/0.5/1.0'
echo 'START: face_blind_spd_oracle_admission'
CUDA_VISIBLE_DEVICES="$physical_gpu" python -u \
  evaluate_face_blind_spd_oracle.py \
  --config "$config" \
  --protocol "$protocol" \
  --baseline-checkpoint "$baseline" \
  --helen-root "$helen_root" \
  --output "$result" \
  --device cuda \
  --eval-bsize "$eval_bsize" \
  | tee "$output/evaluation.log"

set +e
python analyze_face_blind_spd_oracle.py \
  --result "$result" \
  | tee "$output/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ "$analysis_status" -eq 0 ]]; then
  echo 'BLIND-SPD ORACLE PROBE PASSED'
else
  echo 'BLIND-SPD ORACLE PROBE FAILED'
fi
echo 'BLIND-SPD ORACLE PROBE COMPLETED'
