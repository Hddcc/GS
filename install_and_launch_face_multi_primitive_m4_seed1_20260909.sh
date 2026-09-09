#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
implementation_commit=4407bf02fb613bc0abe2b924dc1895e2b89dca0a
physical_gpus="${PHYSICAL_GPUS:-0,2,4}"
admission_gpu="${ADMISSION_GPU:-4}"
evaluation_gpu="${EVALUATION_GPU:-4}"
stamp="$project/results/face_multi_primitive_m4_cuda_admission_20260909.txt"
log="$project/results/face_multi_primitive_m4_seed1_probe_orchestrator.log"
pid_file="$project/results/face_multi_primitive_m4_seed1_probe_orchestrator.pid"

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ -d "$project" ]] || {
  echo "STOP: project directory is missing: $project" >&2
  exit 1
}
IFS=',' read -r -a gpu_array <<< "$physical_gpus"
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
for gpu in "$admission_gpu" "$evaluation_gpu"; do
  [[ -n "${seen_gpus[$gpu]:-}" ]] || {
    echo 'STOP: admission/evaluation GPU must be in PHYSICAL_GPUS' >&2
    exit 1
  }
done
for required in PAYLOAD_SHA256 INSTALLED_SHA256; do
  [[ -f "$bundle_dir/$required" ]] || {
    echo "STOP: bundle manifest is missing: $required" >&2
    exit 1
  }
done
(cd "$bundle_dir" && sha256sum -c PAYLOAD_SHA256)

cd "$project"
previous_log=results/face_msconvstar_seed1_probe_orchestrator.log
[[ -f "$previous_log" ]] && grep -qx \
  'MSCONVSTAR SEED1 PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: MSConvStar probe has not reached a terminal state' >&2
  exit 1
}
renderer_stamp=results/face_memory_efficient_raster_cuda_admission_20260908.txt
[[ -f "$renderer_stamp" ]] && grep -qx \
  'MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED' "$renderer_stamp" || {
  echo 'STOP: memory-efficient renderer admission is missing' >&2
  exit 1
}
for required in \
    save/face_gaussian_baseline_seed1_probe_full5/log.txt \
    save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth; do
  [[ -f "$required" ]] || {
    echo "STOP: missing baseline artifact: $required" >&2
    exit 1
  }
done

check_hash() {
  local file="$1"
  local expected="$2"
  local actual
  [[ -f "$file" ]] || {
    echo "STOP: missing required file: $file" >&2
    exit 1
  }
  actual=$(sha256sum "$file" | cut -d ' ' -f1)
  [[ "$actual" == "$expected" ]] || {
    echo "STOP: hash mismatch: $file" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 1
  }
}
check_hash models/edsr.py \
  47a3288dbf94d481e10ab1370505d0729dbbe3b382de33f4c8acdb7c4e907d0b
check_hash models/gaussian.py \
  e894e96e404fef42b252755f0b63abffee454f0ff13677646c0dac9a6adfc8f3
check_hash models/gaussian_memory_efficient.py \
  9573665bf34ed47503dd9d403269cd77e8bc29804f983724cb46bc4a2ca1c098
check_hash train_gaussian.py \
  ec7bbcd5f68341c2af4eec815ede7adeee186c7a87fe1ffcc292c2a1dc83d1fa

targets=(
  models/gaussian_memory_efficient.py
  models/gaussian_multi_primitive.py
  train_gaussian.py
  configs/train/face/probe_face_gaussian_multi_primitive_m4_seed1.yaml
  validate_face_multi_primitive_configs.py
  smoke_test_face_multi_primitive.py
  evaluate_face_multi_primitive.py
  analyze_face_multi_primitive_probe.py
  test_analyze_face_multi_primitive_probe.py
  test_face_gaussian_multi_primitive.py
  test_train_microbatch_accumulation.py
  run_face_multi_primitive_m4_seed1_probe.sh
  THIRD_PARTY_NOTICES_MULTI_PRIMITIVE.md
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
for path in "$stamp" "$log" "$pid_file" \
    results/face_multi_primitive_m4_seed1_probe \
    save/face_gaussian_multi_primitive_m4_seed1_probe5 \
    results/face_gaussian_multi_primitive_m4_seed1_probe5; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

backup="server_backups/pre_multi_primitive_m4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/models"
cp -a models/__init__.py models/gaussian_memory_efficient.py "$backup/models/"
cp -a train_gaussian.py "$backup/"
for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
if ! grep -qxF 'from . import gaussian_multi_primitive' models/__init__.py; then
  printf '\nfrom . import gaussian_multi_primitive\n' >> models/__init__.py
fi
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")
grep -qxF 'from . import gaussian_multi_primitive' models/__init__.py

python -c 'import torch, yaml'
python -m py_compile \
  models/__init__.py models/gaussian_memory_efficient.py \
  models/gaussian_multi_primitive.py train_gaussian.py \
  validate_face_multi_primitive_configs.py \
  smoke_test_face_multi_primitive.py evaluate_face_multi_primitive.py \
  analyze_face_multi_primitive_probe.py \
  test_analyze_face_multi_primitive_probe.py \
  test_face_gaussian_multi_primitive.py \
  test_train_microbatch_accumulation.py
bash -n run_face_multi_primitive_m4_seed1_probe.sh
python validate_face_multi_primitive_configs.py \
  --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
  --candidate configs/train/face/probe_face_gaussian_multi_primitive_m4_seed1.yaml
python test_train_microbatch_accumulation.py
python test_analyze_face_multi_primitive_probe.py
python test_face_gaussian_multi_primitive.py
python smoke_test_face_multi_primitive.py \
  --config configs/train/face/probe_face_gaussian_multi_primitive_m4_seed1.yaml \
  --device cpu --seed 1 --inp-size 6 --sample-q 41 --batch-size 2
python smoke_test_face_multi_primitive.py \
  --config configs/train/face/probe_face_gaussian_multi_primitive_m4_seed1.yaml \
  --device cuda --gpu "$admission_gpu" --seed 1 \
  --inp-size 16 --sample-q 512 --batch-size 4
{
  echo "IMPLEMENTATION_COMMIT=$implementation_commit"
  echo "PHYSICAL_GPUS=$physical_gpus"
  echo "ADMISSION_GPU=$admission_gpu"
  echo 'MULTI-PRIMITIVE M4 CUDA ADMISSION PASSED'
} | tee "$stamp"

mkdir -p results
nohup env PHYSICAL_GPUS="$physical_gpus" \
  EVALUATION_GPU="$evaluation_gpu" \
  bash run_face_multi_primitive_m4_seed1_probe.sh \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: multi-primitive runner exited during launch' >&2
  tail -n 80 "$log" >&2 || true
  exit 1
fi

echo "MULTI-PRIMITIVE M4 SEED1 PROBE LAUNCHED: pid=$pid"
echo "IMPLEMENTATION_COMMIT=$implementation_commit"
echo "PHYSICAL_GPUS=$physical_gpus"
echo "EVALUATION_GPU=$evaluation_gpu"
echo "LOG=$log"
echo "BACKUP=$backup"
