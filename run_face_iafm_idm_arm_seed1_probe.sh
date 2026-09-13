#!/usr/bin/env bash
set -euo pipefail

cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main

physical_gpus="${PHYSICAL_GPUS:-0,2,4}"
evaluation_gpu="${EVALUATION_GPU:-4}"
IFS=',' read -r -a gpu_array <<< "$physical_gpus"
baseline=save/face_gaussian_baseline_seed1_probe_full5
config=configs/train/face/probe_face_gaussian_iafm_idm_arm_seed1.yaml
name=face_gaussian_iafm_idm_arm_seed1_probe5
run="save/$name"
out="results/$name"
summary=results/face_iafm_idm_arm_seed1_probe

[[ "${#gpu_array[@]}" -eq 3 ]] || {
  echo 'STOP: PHYSICAL_GPUS must contain exactly 0,2,4' >&2
  exit 1
}
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
  [[ "$gpu" =~ ^(0|2|4)$ ]] || {
    echo 'STOP: each physical GPU must be one of 0, 2, and 4' >&2
    exit 1
  }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || {
    echo "STOP: duplicate physical GPU: $gpu" >&2
    exit 1
  }
  seen_gpus[$gpu]=1
done
for gpu in 0 2 4; do
  [[ -n "${seen_gpus[$gpu]:-}" ]] || {
    echo 'STOP: PHYSICAL_GPUS must contain exactly 0,2,4' >&2
    exit 1
  }
done
[[ -n "${seen_gpus[$evaluation_gpu]:-}" ]] || {
  echo 'STOP: EVALUATION_GPU must be included in PHYSICAL_GPUS' >&2
  exit 1
}
previous_log=results/face_blind_spd_oracle_admission_orchestrator.log
[[ -f "$previous_log" ]] \
  && grep -qx 'BLIND-SPD ORACLE PROBE FAILED' "$previous_log" \
  && grep -qx 'BLIND-SPD ORACLE PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: blind-SPD oracle does not have the expected failed terminal state' >&2
  exit 1
}
for required in "$baseline/log.txt" "$baseline/epoch-5.pth"; do
  [[ -f "$required" ]] || {
    echo "STOP: missing baseline artifact: $required" >&2
    exit 1
  }
done
for path in "$run" "$out" "$summary"; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

mkdir -p "$out" "$summary"
echo "RUNTIME: physical GPUs $physical_gpus, DataParallel 4 x 3, batch 12"
echo "EVALUATION: physical GPU $evaluation_gpu"
python validate_face_iafm_idm_arm_configs.py \
  --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
  --candidate "$config" \
  --expected-seed 1
echo "START: $name"
python -u train_gaussian.py \
  --config "$config" \
  --name "$name" \
  --gpu "$physical_gpus" \
  --seed 1 \
  > "$out/train_console.log" 2>&1
if grep -qiE 'nan|out of memory|traceback|runtimeerror' \
    "$out/train_console.log"; then
  echo 'STOP: failure pattern found in IAFM training' >&2
  exit 1
fi
for required in "$run/log.txt" "$run/epoch-5.pth"; do
  [[ -f "$required" ]] || {
    echo "STOP: missing candidate artifact: $required" >&2
    exit 1
  }
done
grep -nE 'epoch .*val:' "$run/log.txt"

CUDA_VISIBLE_DEVICES="$evaluation_gpu" python -u \
  evaluate_face_iafm_idm_arm.py \
  --config "$config" \
  --baseline-checkpoint "$baseline/epoch-5.pth" \
  --candidate-checkpoint "$run/epoch-5.pth" \
  --device cuda \
  --eval-bsize 10000 \
  --max-samples 100 \
  | tee "$summary/evaluation.log"
echo "DONE: $name"

set +e
python analyze_face_iafm_idm_arm_probe.py \
  --baseline-log "$baseline/log.txt" \
  --candidate-log "$run/log.txt" \
  --evaluation-log "$summary/evaluation.log" \
  | tee "$summary/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ "$analysis_status" -eq 0 ]]; then
  echo 'IAFM IDM+ARM SEED1 PROBE PASSED'
else
  echo 'IAFM IDM+ARM SEED1 PROBE FAILED'
fi
echo 'IAFM IDM+ARM SEED1 PROBE COMPLETED'
