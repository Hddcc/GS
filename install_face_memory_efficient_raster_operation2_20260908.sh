#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
source_commit=da41c9e17e26d7a015de2df5af0efcfd38b4a0df
physical_gpu="${PHYSICAL_GPU:-4}"
stamp="$project/results/face_memory_efficient_raster_cuda_admission_20260908.txt"
benchmark_log="$project/results/face_memory_efficient_raster_cuda_screen_20260908.log"

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ -d "$project" ]] || {
  echo "STOP: project directory is missing: $project" >&2
  exit 1
}
[[ "$physical_gpu" =~ ^(0|2|4)$ ]] || {
  echo 'STOP: PHYSICAL_GPU must be one of the currently available cards: 0, 2, 4' >&2
  exit 1
}
for required in PAYLOAD_SHA256 INSTALLED_SHA256; do
  [[ -f "$bundle_dir/$required" ]] || {
    echo "STOP: bundle manifest is missing: $required" >&2
    exit 1
  }
done
(cd "$bundle_dir" && sha256sum -c PAYLOAD_SHA256)

cd "$project"
previous_log=results/face_continuous_spd_staged_probe_orchestrator.log
if [[ ! -f "$previous_log" ]] || ! grep -qE \
    'CONTINUOUS-SPD STAGED PROBE (STOPPED AFTER SEED 1|COMPLETED)' \
    "$previous_log"; then
  echo 'STOP: operation 1 has not reached a terminal state' >&2
  exit 1
fi
baseline_checkpoint=save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth
[[ -f "$baseline_checkpoint" ]] || {
  echo "STOP: baseline checkpoint is missing: $baseline_checkpoint" >&2
  exit 1
}

if [[ -f "$stamp" ]] && grep -qx \
    'MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED' "$stamp"; then
  (cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")
  echo "ALREADY PASSED: $stamp"
  exit 0
fi

targets=(
  models/gaussian_memory_efficient.py
  configs/train/face/probe_face_gaussian_memory_efficient_seed1.yaml
  test_memory_efficient_raster.py
  benchmark_memory_efficient_raster.py
  validate_face_memory_efficient_config.py
)
for target in "${targets[@]}"; do
  [[ -f "$payload/$target" ]] || {
    echo "STOP: payload file is missing: $target" >&2
    exit 1
  }
  if [[ -e "$target" ]]; then
    installed_hash=$(sha256sum "$target" | cut -d ' ' -f1)
    payload_hash=$(sha256sum "$payload/$target" | cut -d ' ' -f1)
    [[ "$installed_hash" == "$payload_hash" ]] || {
      echo "STOP: target exists with different content: $target" >&2
      echo "installed=$installed_hash" >&2
      echo "payload=$payload_hash" >&2
      exit 1
    }
  fi
done

backup="server_backups/pre_memory_efficient_raster_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/models"
cp -a models/__init__.py "$backup/models/"
for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
if ! grep -qxF 'from . import gaussian_memory_efficient' models/__init__.py; then
  printf '\nfrom . import gaussian_memory_efficient\n' >> models/__init__.py
fi
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")

python -m py_compile \
  models/__init__.py models/gaussian_memory_efficient.py \
  test_memory_efficient_raster.py benchmark_memory_efficient_raster.py \
  validate_face_memory_efficient_config.py
python validate_face_memory_efficient_config.py \
  --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
  --candidate configs/train/face/probe_face_gaussian_memory_efficient_seed1.yaml
python test_memory_efficient_raster.py --device cpu --seed 1
python test_memory_efficient_raster.py \
  --device cuda --gpu "$physical_gpu" --seed 1
mkdir -p results
python benchmark_memory_efficient_raster.py \
  --checkpoint "$baseline_checkpoint" --gpu "$physical_gpu" --seed 1 \
  --input-height 16 --input-width 24 --scale 8 --queries 512 \
  --warmup 1 --repeats 3 --minimum-memory-reduction 0.30 \
  --maximum-output-error 2e-5 | tee "$benchmark_log"

{
  echo "SOURCE_COMMIT=$source_commit"
  echo "PHYSICAL_GPU=$physical_gpu"
  echo "BENCHMARK_LOG=$benchmark_log"
  echo 'MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED'
} | tee "$stamp"
echo "BACKUP=$backup"
echo 'STOP HERE: send the CUDA screen output back before channel-budget training.'
