#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
trap 'status=$?; if (( status != 0 )); then echo "SCALE-GATED FREQUENCY PROBE FAILED: exit=$status"; fi; rmdir .launch_lock 2>/dev/null || true' EXIT
IFS=, read -r -a gpus <<< "${PHYSICAL_GPUS:-3,4}"
gpu0="${gpus[0]}"
gpu1="${gpus[1]:-${gpus[0]}}"
echo "RUNTIME: control GPU $gpu0; residual/gated GPU $gpu1"
echo 'RUNTIME: microbatch 4 x 3, effective batch 12; fixed final checkpoint'
echo 'PROTOCOL: exploratory CelebA validation; scales 2,2.5,3,3.5,4'
train_variant() {
    local variant="$1" gpu="$2"
    mkdir -p "results/$variant"
    echo "START: $variant GPU=$gpu"
    python -u train_gaussian.py \
        --config "configs/generated/scale_gated_frequency/$variant.yaml" \
        --name "face_scale_gated_frequency_${variant}_seed1" \
        --gpu "$gpu" --microbatch-size 4 \
        > "results/$variant/train_console.log" 2>&1
    echo "DONE: $variant"
}
if [[ "$gpu0" == "$gpu1" ]]; then
    train_variant control "$gpu0"
    train_variant residual "$gpu1"
else
    train_variant control "$gpu0" &
    control_pid=$!
    train_variant residual "$gpu1" &
    residual_pid=$!
    failed=0
    wait "$control_pid" || failed=1
    wait "$residual_pid" || failed=1
    (( failed == 0 )) || exit 1
fi
train_variant gated "$gpu1"
CUDA_VISIBLE_DEVICES="${EVALUATION_GPU:-$gpu1}" python -u \
    evaluate_face_scale_gated_frequency.py --samples "${EVAL_SAMPLES:-100}" \
    > results/evaluation.log 2>&1
cat results/evaluation.log
echo 'SCALE-GATED FREQUENCY SEED1 PROBE COMPLETED'
