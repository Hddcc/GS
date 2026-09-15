#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
implementation_commit=bdec36133ba92c91d7dcf4d096192808117afc46
physical_gpus="${PHYSICAL_GPUS:-0,1}"
admission_gpu="${ADMISSION_GPU:-1}"
stamp="$project/results/face_gsasr_zero_train_cuda_admission_20260915.txt"
log="$project/results/face_gsasr_zero_train_admission_orchestrator.log"
pid_file="$project/results/face_gsasr_zero_train_admission_orchestrator.pid"
result_dir="$project/results/face_gsasr_zero_train_admission"
official_dir=third_party/gsasr_paper_9d2eb64

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ -d "$project" ]] || {
  echo "STOP: project directory is missing: $project" >&2
  exit 1
}
IFS=',' read -r -a gpu_array <<< "$physical_gpus"
[[ "${#gpu_array[@]}" -eq 2 || "${#gpu_array[@]}" -eq 3 ]] || {
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
[[ -n "${seen_gpus[$admission_gpu]:-}" ]] || {
  echo 'STOP: ADMISSION_GPU must be present in PHYSICAL_GPUS' >&2
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
previous_log=results/face_iafm_idm_arm_seed1_probe_orchestrator.log
[[ -f "$previous_log" ]] \
  && grep -qx 'IAFM IDM+ARM SEED1 PROBE FAILED' "$previous_log" \
  && grep -qx 'IAFM IDM+ARM SEED1 PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: IAFM IDM+ARM does not have the expected failed terminal state' >&2
  exit 1
}
for required in \
    save/face_gaussian_baseline_seed1_probe_full5/log.txt \
    save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth \
    configs/train/face/probe_face_gaussian_baseline_seed1.yaml; do
  [[ -f "$required" ]] || {
    echo "STOP: missing baseline artifact: $required" >&2
    exit 1
  }
done
data_root=Dataset/FACE/CelebA/val/HR
[[ -d "$data_root" ]] || {
  echo "STOP: missing dataset directory: $data_root" >&2
  exit 1
}
image_count=$(find "$data_root" -maxdepth 1 -type f | wc -l)
[[ "$image_count" -ge 100 ]] || {
  echo "STOP: CelebA val requires at least 100 files; found $image_count" >&2
  exit 1
}
command -v nvcc >/dev/null 2>&1 || {
  echo 'STOP: nvcc is required to compile the official GSASR renderer' >&2
  exit 1
}
python -c 'import PIL, torch, torchvision, yaml; from torch.utils.cpp_extension import CUDA_HOME; assert CUDA_HOME'

encoder_hash=a61351983470a6bbda0691980cfcba0d3b75a07305c2a14000ce663a49e54847
decoder_hash=b96999739e6d5f6a81ef9aa432ee2350ed21b205bc0e6b98e3a40613f6898d87
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
check_hash "$payload/$official_dir/weights/encoder.pth" "$encoder_hash"
check_hash "$payload/$official_dir/weights/decoder.pth" "$decoder_hash"

targets=(
  evaluate_face_gsasr_baseline.py
  analyze_face_gsasr_zero_train.py
  test_analyze_face_gsasr_zero_train.py
  test_face_gsasr_official_subset.py
  run_face_gsasr_zero_train_admission.sh
  README_face_gsasr_zero_train_admission_20260915.txt
  THIRD_PARTY_NOTICES_GSASR.md
  "$official_dir/LICENSE"
  "$official_dir/setup_gscuda.py"
  "$official_dir/evaluate_face_gsasr_candidate.py"
  "$official_dir/smoke_test_face_gsasr_cuda.py"
  "$official_dir/utils/__init__.py"
  "$official_dir/utils/edsrbaseline.py"
  "$official_dir/utils/fea2gs.py"
  "$official_dir/utils/gaussian_splatting.py"
  "$official_dir/utils/gs_cuda_dmax/__init__.py"
  "$official_dir/utils/gs_cuda_dmax/gswrapper.py"
  "$official_dir/utils/gs_cuda_dmax/gswrapper.cpp"
  "$official_dir/utils/gs_cuda_dmax/gs.h"
  "$official_dir/utils/gs_cuda_dmax/gs.cu"
  "$official_dir/weights/encoder.pth"
  "$official_dir/weights/decoder.pth"
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
for path in "$stamp" "$log" "$pid_file" "$result_dir"; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")
check_hash "$official_dir/weights/encoder.pth" "$encoder_hash"
check_hash "$official_dir/weights/decoder.pth" "$decoder_hash"

python -m py_compile \
  evaluate_face_gsasr_baseline.py analyze_face_gsasr_zero_train.py \
  test_analyze_face_gsasr_zero_train.py test_face_gsasr_official_subset.py \
  "$official_dir/evaluate_face_gsasr_candidate.py" \
  "$official_dir/smoke_test_face_gsasr_cuda.py" \
  "$official_dir/utils/edsrbaseline.py" "$official_dir/utils/fea2gs.py" \
  "$official_dir/utils/gaussian_splatting.py" \
  "$official_dir/utils/gs_cuda_dmax/gswrapper.py"
bash -n run_face_gsasr_zero_train_admission.sh
python test_analyze_face_gsasr_zero_train.py
python test_face_gsasr_official_subset.py \
  --weights "$official_dir/weights"
(
  cd "$official_dir"
  python setup_gscuda.py build_ext --inplace --force
  CUDA_VISIBLE_DEVICES="$admission_gpu" python smoke_test_face_gsasr_cuda.py \
    --weights weights
)
{
  echo "IMPLEMENTATION_COMMIT=$implementation_commit"
  echo "OFFICIAL_COMMIT=9d2eb64a51303ce7a22fd672197185488b38e10a"
  echo "PHYSICAL_GPUS=$physical_gpus"
  echo "ADMISSION_GPU=$admission_gpu"
  echo 'GSASR ZERO-TRAIN CUDA ADMISSION PASSED'
} | tee "$stamp"

mkdir -p results
nohup env PHYSICAL_GPUS="$physical_gpus" \
  bash run_face_gsasr_zero_train_admission.sh \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: GSASR zero-train runner exited during launch' >&2
  tail -n 100 "$log" >&2 || true
  exit 1
fi

echo "GSASR ZERO-TRAIN ADMISSION LAUNCHED: pid=$pid"
echo "IMPLEMENTATION_COMMIT=$implementation_commit"
echo "PHYSICAL_GPUS=$physical_gpus"
echo "LOG=$log"
