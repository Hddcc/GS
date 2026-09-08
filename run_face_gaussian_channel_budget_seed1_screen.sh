#!/usr/bin/env bash
set -euo pipefail

cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main

physical_gpus="${PHYSICAL_GPUS:-0,2,4}"
IFS=',' read -r -a gpu_array <<< "$physical_gpus"
budgets=(16 32 64)
microbatch_size=4
summary=results/face_gaussian_channel_budget_seed1_screen
baseline=save/face_gaussian_baseline_seed1_probe_full5
admission=results/face_memory_efficient_raster_cuda_admission_20260908.txt

if [[ "${#gpu_array[@]}" -ne 3 ]]; then
  echo 'STOP: PHYSICAL_GPUS must contain exactly three comma-separated GPUs' >&2
  exit 1
fi
declare -A seen_gpus=()
for gpu in "${gpu_array[@]}"; do
  [[ "$gpu" =~ ^(0|2|4)$ ]] || {
    echo 'STOP: each GPU must be one of the available cards: 0, 2, 4' >&2
    exit 1
  }
  [[ -z "${seen_gpus[$gpu]:-}" ]] || {
    echo "STOP: duplicate GPU in PHYSICAL_GPUS: $gpu" >&2
    exit 1
  }
  seen_gpus[$gpu]=1
done
[[ -f "$admission" ]] || {
  echo "STOP: operation-2 CUDA admission stamp is missing: $admission" >&2
  exit 1
}
grep -qx 'MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED' "$admission" || {
  echo 'STOP: operation-2 CUDA admission has not passed' >&2
  exit 1
}
for required in "$baseline/log.txt" "$baseline/epoch-5.pth"; do
  [[ -f "$required" ]] || {
    echo "STOP: missing baseline artifact: $required" >&2
    exit 1
  }
done
[[ ! -e "$summary" ]] || {
  echo "STOP: summary path exists; do not overwrite: $summary" >&2
  exit 1
}
for budget in "${budgets[@]}"; do
  name="face_gaussian_channel_budget_cg${budget}_seed1_probe5"
  for path in "save/$name" "results/$name"; do
    [[ ! -e "$path" ]] || {
      echo "STOP: candidate path exists; do not overwrite: $path" >&2
      exit 1
    }
  done
done

mkdir -p "$summary"
echo "RUNTIME: Cg16/Cg32/Cg64 on physical GPUs ${physical_gpus}"
echo 'RUNTIME: microbatch 4 x 3, effective batch 12 per candidate'
echo 'REFERENCE: Cg8 reuses the mathematically equivalent seed-1 baseline'

run_candidate() {
  local budget="$1"
  local gpu="$2"
  local config="configs/train/face/probe_face_gaussian_channel_budget_cg${budget}_seed1.yaml"
  local name="face_gaussian_channel_budget_cg${budget}_seed1_probe5"
  local run="save/$name"
  local out="results/$name"
  local evaluation_log="$summary/evaluation_cg${budget}.log"

  python validate_face_gaussian_channel_budget_configs.py \
    --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
    --candidate "$config" \
    --expected-channels "$budget"
  mkdir -p "$out"
  echo "START: $name on physical GPU $gpu"
  python -u train_gaussian.py \
    --config "$config" \
    --name "$name" \
    --gpu "$gpu" \
    --seed 1 \
    --microbatch-size "$microbatch_size" \
    > "$out/train_console.log" 2>&1
  if grep -qiE 'nan|out of memory|traceback|runtimeerror' \
      "$out/train_console.log"; then
    echo "STOP: failure pattern found for Cg=$budget" >&2
    return 1
  fi
  for required in "$run/log.txt" "$run/epoch-5.pth"; do
    [[ -f "$required" ]] || {
      echo "STOP: missing candidate artifact: $required" >&2
      return 1
    }
  done
  grep -nE 'epoch .*val:' "$run/log.txt"
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    evaluate_face_gaussian_channel_budget.py \
    --config "$config" \
    --baseline-checkpoint "$baseline/epoch-5.pth" \
    --candidate-checkpoint "$run/epoch-5.pth" \
    --device cuda \
    --eval-bsize 10000 \
    --max-samples 100 \
    | tee "$evaluation_log"
  echo "DONE: $name"
}

pids=()
for index in 0 1 2; do
  budget="${budgets[$index]}"
  gpu="${gpu_array[$index]}"
  run_candidate "$budget" "$gpu" \
    > "$summary/worker_cg${budget}.log" 2>&1 &
  pids+=("$!")
done

cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

failed=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    failed=1
  fi
done
trap - INT TERM
for budget in "${budgets[@]}"; do
  echo "--- Cg=${budget} worker summary ---"
  tail -n 40 "$summary/worker_cg${budget}.log"
done
if [[ "$failed" -ne 0 ]]; then
  echo 'GAUSSIAN CHANNEL-BUDGET SCREEN FAILED DURING A WORKER' >&2
  exit 1
fi

set +e
python analyze_face_gaussian_channel_budget.py \
  --baseline-log "$baseline/log.txt" \
  --candidate 16 \
    save/face_gaussian_channel_budget_cg16_seed1_probe5/log.txt \
    "$summary/evaluation_cg16.log" \
  --candidate 32 \
    save/face_gaussian_channel_budget_cg32_seed1_probe5/log.txt \
    "$summary/evaluation_cg32.log" \
  --candidate 64 \
    save/face_gaussian_channel_budget_cg64_seed1_probe5/log.txt \
    "$summary/evaluation_cg64.log" \
  | tee "$summary/analysis.log"
analysis_status=${PIPESTATUS[0]}
set -e
if [[ "$analysis_status" -ne 0 ]]; then
  echo 'GAUSSIAN CHANNEL-BUDGET SEED1 SCREEN FAILED'
fi
echo 'GAUSSIAN CHANNEL-BUDGET SEED1 SCREEN COMPLETED'
