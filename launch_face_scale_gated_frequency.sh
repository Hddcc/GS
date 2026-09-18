#!/usr/bin/env bash
set -Eeuo pipefail
bundle="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$PWD}"
checkpoint="${BASELINE_CHECKPOINT:-$project_root/save/face_gaussian_baseline_seed1_formal_v2protocol/epoch-best.pth}"
cd "$bundle"
sha256sum --check FILES_SHA256
cd runtime
[[ -f "$checkpoint" ]] || { echo "STOP: formal baseline checkpoint not found: $checkpoint"; exit 1; }
[[ ! -d save ]] || { echo 'STOP: an experiment already exists; preserve it and use a fresh extracted bundle'; exit 1; }
mkdir .launch_lock || { echo 'STOP: this bundle is already running or awaiting recovery'; exit 1; }
launched=0
trap 'if (( launched == 0 )); then rmdir .launch_lock 2>/dev/null || true; fi' EXIT
IFS=, read -r -a gpus <<< "${PHYSICAL_GPUS:-3,4}"
for gpu in "${gpus[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "STOP: invalid GPU index: $gpu"; exit 1; }
done
admission_gpu="${ADMISSION_GPU:-${gpus[${#gpus[@]}-1]}}"
CUDA_VISIBLE_DEVICES="$admission_gpu" python -u test_face_scale_gated_frequency.py --device cuda
python -u prepare_face_scale_gated_frequency.py --project-root "$project_root" \
    --checkpoint "$checkpoint" --epochs "${EPOCHS:-30}"
PHYSICAL_GPUS="${PHYSICAL_GPUS:-3,4}" nohup bash run_face_scale_gated_frequency.sh \
    > results/orchestrator.log 2>&1 < /dev/null &
pid=$!
launched=1
echo "SCALE-GATED FREQUENCY PROBE LAUNCHED: pid=$pid"
echo "LOG=$bundle/runtime/results/orchestrator.log"
echo "TRAIN_LOG=$bundle/runtime/results/gated/train_console.log"
echo "IMPLEMENTATION_COMMIT=$(tr -d '\r\n' < "$bundle/IMPLEMENTATION_COMMIT")"
