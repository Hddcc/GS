#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
physical_gpu="${PHYSICAL_GPU:-4}"
run_name=face_gaussian_frequency_residual_scale_edge_seed1_probe5
log="$project/results/face_scale_conditioned_edge_probe_orchestrator.log"
pid_file="$project/results/face_scale_conditioned_edge_probe_orchestrator.pid"

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ "$physical_gpu" =~ ^[0-9]+$ ]] || {
  echo 'STOP: PHYSICAL_GPU must be one CUDA device index' >&2
  exit 1
}
[[ -d "$project" ]] || {
  echo "STOP: project directory is missing: $project" >&2
  exit 1
}
[[ -f "$bundle_dir/PAYLOAD_SHA256" ]] || {
  echo 'STOP: PAYLOAD_SHA256 is missing' >&2
  exit 1
}
normalized_manifest=$(mktemp)
trap 'rm -f "$normalized_manifest"' EXIT
tr -d '\r' < "$bundle_dir/PAYLOAD_SHA256" > "$normalized_manifest"
(cd "$bundle_dir" && sha256sum -c "$normalized_manifest")

targets=(
  datasets/wrappers.py
  configs/train/face/probe_face_gaussian_frequency_residual_scale_edge_seed1.yaml
  test_scale_conditioned_edge_weight.py
)
for target in "${targets[@]}"; do
  [[ -f "$payload/$target" ]] || {
    echo "STOP: payload file is missing: $target" >&2
    exit 1
  }
done

for required in \
    train_gaussian.py \
    models/gaussian_face_frequency_residual.py \
    datasets/__init__.py \
    utils.py; do
  [[ -f "$project/$required" ]] || {
    echo "STOP: required project file is missing: $required" >&2
    exit 1
  }
done

for path in "$log" "$pid_file" "$project/save/$run_name"; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

backup="$project/server_backups/pre_scale_conditioned_edge_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/datasets" "$backup/configs/train/face"
cp -a "$project/datasets/wrappers.py" "$backup/datasets/"
cp -a "$project/configs/train/face/probe_face_gaussian_frequency_residual_scale_edge_seed1.yaml" \
  "$backup/configs/train/face/" 2>/dev/null || true
cp -a "$project/test_scale_conditioned_edge_weight.py" "$backup/" 2>/dev/null || true

for target in "${targets[@]}"; do
  mkdir -p "$project/$(dirname "$target")"
  cp -a "$payload/$target" "$project/$target"
done

cd "$project"
python -c 'import torch, yaml, imageio'
python -m py_compile datasets/wrappers.py test_scale_conditioned_edge_weight.py
python test_scale_conditioned_edge_weight.py

mkdir -p results
nohup env PHYSICAL_GPU="$physical_gpu" RUN_NAME="$run_name" \
  bash "$bundle_dir/run_face_scale_conditioned_edge_probe_20260919.sh" \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: scale-conditioned edge probe exited during launch' >&2
  tail -n 100 "$log" >&2 || true
  exit 1
fi

echo "SCALE-CONDITIONED EDGE PROBE LAUNCHED: pid=$pid"
echo "PHYSICAL_GPU=$physical_gpu"
echo "LOG=$log"
echo "BACKUP=$backup"
