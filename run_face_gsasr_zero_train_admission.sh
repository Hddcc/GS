#!/usr/bin/env bash
set -uo pipefail

cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main

physical_gpus="${PHYSICAL_GPUS:-0,1}"
eval_bsize="${EVAL_BSIZE:-10000}"
baseline=save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth
config=configs/train/face/probe_face_gaussian_baseline_seed1.yaml
data_root=Dataset/FACE/CelebA/val/HR
third_party=third_party/gsasr_paper_9d2eb64
weights="$third_party/weights"
output=results/face_gsasr_zero_train_admission
completed=0

finish_on_error() {
  local status=$?
  if [[ "$status" -ne 0 && "$completed" -eq 0 ]]; then
    echo 'GSASR ZERO-TRAIN ADMISSION FAILED'
    echo 'GSASR ZERO-TRAIN ADMISSION COMPLETED'
  fi
  exit "$status"
}
trap finish_on_error EXIT

IFS=',' read -r -a gpu_array <<< "$physical_gpus"
gpu_count="${#gpu_array[@]}"
[[ "$gpu_count" -eq 2 || "$gpu_count" -eq 3 ]] || {
  echo 'STOP: PHYSICAL_GPUS must contain two or three GPU ids' >&2
  exit 1
}
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo 'STOP: every physical GPU id must be a non-negative integer' >&2
    exit 1
  }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || {
    echo "STOP: duplicate physical GPU id: $gpu" >&2
    exit 1
  }
  seen_gpus[$gpu]=1
done
[[ "$eval_bsize" =~ ^[1-9][0-9]*$ ]] || {
  echo 'STOP: EVAL_BSIZE must be a positive integer' >&2
  exit 1
}
for required in "$baseline" "$config" "$weights/encoder.pth" \
    "$weights/decoder.pth" "$third_party/evaluate_face_gsasr_candidate.py"; do
  [[ -f "$required" ]] || {
    echo "STOP: missing required file: $required" >&2
    exit 1
  }
done
[[ -d "$data_root" ]] || {
  echo "STOP: missing dataset directory: $data_root" >&2
  exit 1
}
[[ ! -e "$output" ]] || {
  echo "STOP: output already exists; do not overwrite: $output" >&2
  exit 1
}

mkdir -p "$output"
baseline_gpu="${gpu_array[$((gpu_count - 1))]}"
echo "RUNTIME: GSASR x2/x4/x8 in waves on physical GPUs $physical_gpus"
echo "BASELINE: physical GPU $baseline_gpu, paired 100-image RGB protocol"
echo 'MODE: official GSASR paper EDSR weights, evaluation only, no training'
echo 'START: face_gsasr_zero_train_admission'

CUDA_VISIBLE_DEVICES="$baseline_gpu" python -u evaluate_face_gsasr_baseline.py \
  --config "$config" \
  --checkpoint "$baseline" \
  --output "$output/baseline.json" \
  --max-samples 100 \
  --eval-bsize "$eval_bsize" \
  | tee "$output/baseline.log"
baseline_status=${PIPESTATUS[0]}
[[ "$baseline_status" -eq 0 ]] || exit "$baseline_status"

scales=(2 4 8)
candidate_status=0
for ((wave_start = 0; wave_start < 3; wave_start += gpu_count)); do
  wave_pids=()
  wave_scales=()
  for ((slot = 0; slot < gpu_count; slot++)); do
    index=$((wave_start + slot))
    [[ "$index" -lt 3 ]] || break
    scale="${scales[$index]}"
    gpu="${gpu_array[$slot]}"
    (
      cd "$third_party" || exit 1
      CUDA_VISIBLE_DEVICES="$gpu" python -u evaluate_face_gsasr_candidate.py \
        --baseline-json "../../$output/baseline.json" \
        --data-root "../../$data_root" \
        --weights weights \
        --output "../../$output/candidate_x${scale}.json" \
        --scale "$scale" \
        --max-samples 100 \
        2>&1 \
        | sed -u "s/^/[x${scale}] /" \
        | tee "../../$output/candidate_x${scale}.log"
      exit "${PIPESTATUS[0]}"
    ) &
    pid=$!
    wave_pids+=("$pid")
    wave_scales+=("$scale")
    echo "LAUNCHED: GSASR x$scale on physical GPU $gpu, pid=$pid"
  done
  for slot in "${!wave_pids[@]}"; do
    if ! wait "${wave_pids[$slot]}"; then
      scale="${wave_scales[$slot]}"
      echo "ERROR: GSASR x$scale candidate evaluation failed" >&2
      tail -n 80 "$output/candidate_x${scale}.log" >&2 || true
      candidate_status=1
    fi
  done
  [[ "$candidate_status" -eq 0 ]] || break
done
[[ "$candidate_status" -eq 0 ]] || exit 1

set +e
python analyze_face_gsasr_zero_train.py \
  --baseline-json "$output/baseline.json" \
  --candidate-json \
    "$output/candidate_x2.json" \
    "$output/candidate_x4.json" \
    "$output/candidate_x8.json" \
  --expected-samples 100 \
  | tee "$output/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ "$analysis_status" -eq 0 ]]; then
  echo 'GSASR ZERO-TRAIN PROBE PASSED'
else
  echo 'GSASR ZERO-TRAIN PROBE FAILED'
fi
echo 'GSASR ZERO-TRAIN ADMISSION COMPLETED'
completed=1
trap - EXIT
exit "$analysis_status"
