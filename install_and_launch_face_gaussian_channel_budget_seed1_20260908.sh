#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
implementation_commit=004d2d3260e308dfda82441d85805df2e8594ff8
physical_gpus="${PHYSICAL_GPUS:-0,2,4}"
log="$project/results/face_gaussian_channel_budget_seed1_screen_orchestrator.log"
pid_file="$project/results/face_gaussian_channel_budget_seed1_screen_orchestrator.pid"

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ -d "$project" ]] || {
  echo "STOP: project directory is missing: $project" >&2
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
admission=results/face_memory_efficient_raster_cuda_admission_20260908.txt
[[ -f "$admission" ]] || {
  echo "STOP: operation-2 CUDA admission stamp is missing: $admission" >&2
  exit 1
}
grep -qx 'MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED' "$admission" || {
  echo 'STOP: operation-2 CUDA admission has not passed' >&2
  exit 1
}

targets=(
  models/gaussian_memory_efficient.py
  train_gaussian.py
  analyze_face_gaussian_channel_budget.py
  evaluate_face_gaussian_channel_budget.py
  validate_face_gaussian_channel_budget_configs.py
  test_analyze_face_gaussian_channel_budget.py
  test_face_gaussian_channel_budget.py
  test_train_microbatch_accumulation.py
  run_face_gaussian_channel_budget_seed1_screen.sh
  configs/train/face/probe_face_gaussian_channel_budget_cg16_seed1.yaml
  configs/train/face/probe_face_gaussian_channel_budget_cg32_seed1.yaml
  configs/train/face/probe_face_gaussian_channel_budget_cg64_seed1.yaml
)
for target in "${targets[@]}"; do
  [[ -f "$payload/$target" ]] || {
    echo "STOP: payload file is missing: $target" >&2
    exit 1
  }
done

old_model_hash=99326efadbcfb494a704d9efcff94f4499a86dd4509ef058a35e454675919ef6
new_model_hash=9573665bf34ed47503dd9d403269cd77e8bc29804f983724cb46bc4a2ca1c098
train_hash=ec7bbcd5f68341c2af4eec815ede7adeee186c7a87fe1ffcc292c2a1dc83d1fa
installed_model_hash=$(sha256sum models/gaussian_memory_efficient.py | cut -d ' ' -f1)
[[ "$installed_model_hash" == "$old_model_hash" \
    || "$installed_model_hash" == "$new_model_hash" ]] || {
  echo 'STOP: installed memory-efficient model is not an admitted version' >&2
  echo "actual=$installed_model_hash" >&2
  exit 1
}
installed_train_hash=$(sha256sum train_gaussian.py | cut -d ' ' -f1)
[[ "$installed_train_hash" == "$train_hash" ]] || {
  echo 'STOP: train_gaussian.py does not match the validated microbatch version' >&2
  echo "actual=$installed_train_hash" >&2
  exit 1
}
for target in "${targets[@]:2}"; do
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
for path in "$log" "$pid_file" results/face_gaussian_channel_budget_seed1_screen; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

backup="server_backups/pre_gaussian_channel_budget_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/models"
cp -a models/__init__.py models/gaussian_memory_efficient.py "$backup/models/"
cp -a train_gaussian.py "$backup/"
for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
if ! grep -qxF 'from . import gaussian_memory_efficient' models/__init__.py; then
  printf '\nfrom . import gaussian_memory_efficient\n' >> models/__init__.py
fi
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")

python -m py_compile \
  models/__init__.py models/gaussian_memory_efficient.py train_gaussian.py \
  analyze_face_gaussian_channel_budget.py \
  evaluate_face_gaussian_channel_budget.py \
  validate_face_gaussian_channel_budget_configs.py \
  test_analyze_face_gaussian_channel_budget.py \
  test_face_gaussian_channel_budget.py \
  test_train_microbatch_accumulation.py
bash -n run_face_gaussian_channel_budget_seed1_screen.sh
python test_train_microbatch_accumulation.py
python test_analyze_face_gaussian_channel_budget.py
python test_face_gaussian_channel_budget.py
for channels in 16 32 64; do
  python validate_face_gaussian_channel_budget_configs.py \
    --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
    --candidate \
      "configs/train/face/probe_face_gaussian_channel_budget_cg${channels}_seed1.yaml" \
    --expected-channels "$channels"
done

mkdir -p results
nohup env PHYSICAL_GPUS="$physical_gpus" \
  bash run_face_gaussian_channel_budget_seed1_screen.sh \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: channel-budget runner exited during launch' >&2
  tail -n 80 "$log" >&2 || true
  exit 1
fi

echo "GAUSSIAN CHANNEL-BUDGET SEED1 SCREEN LAUNCHED: pid=$pid"
echo "IMPLEMENTATION_COMMIT=$implementation_commit"
echo "PHYSICAL_GPUS=$physical_gpus"
echo "LOG=$log"
echo "BACKUP=$backup"
