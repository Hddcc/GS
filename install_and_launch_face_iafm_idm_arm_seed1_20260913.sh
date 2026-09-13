#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
implementation_commit=e4b637e584605e68828018330201a4b1d064a0d1
physical_gpus="${PHYSICAL_GPUS:-0,2,4}"
admission_gpu="${ADMISSION_GPU:-4}"
evaluation_gpu="${EVALUATION_GPU:-4}"
stamp="$project/results/face_iafm_idm_arm_cuda_admission_20260913.txt"
log="$project/results/face_iafm_idm_arm_seed1_probe_orchestrator.log"
pid_file="$project/results/face_iafm_idm_arm_seed1_probe_orchestrator.pid"

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
previous_log=results/face_blind_spd_oracle_admission_orchestrator.log
[[ -f "$previous_log" ]] \
  && grep -qx 'BLIND-SPD ORACLE PROBE FAILED' "$previous_log" \
  && grep -qx 'BLIND-SPD ORACLE PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: blind-SPD oracle does not have the expected failed terminal state' >&2
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

check_preimage_or_payload() {
  local file="$1"
  local preimage="$2"
  local actual payload_hash
  [[ -f "$file" ]] || {
    echo "STOP: missing required file: $file" >&2
    exit 1
  }
  actual=$(sha256sum "$file" | cut -d ' ' -f1)
  payload_hash=$(sha256sum "$payload/$file" | cut -d ' ' -f1)
  [[ "$actual" == "$preimage" || "$actual" == "$payload_hash" ]] || {
    echo "STOP: hash mismatch: $file" >&2
    echo "expected-preimage=$preimage" >&2
    echo "expected-payload=$payload_hash" >&2
    echo "actual=$actual" >&2
    exit 1
  }
}

check_hash models/edsr.py \
  47a3288dbf94d481e10ab1370505d0729dbbe3b382de33f4c8acdb7c4e907d0b
check_preimage_or_payload models/__init__.py \
  c8492d96fbddb22521b36311de3599be8089cd707b79729daf22183f86ce0b3d
check_preimage_or_payload models/gaussian.py \
  c4de21d79135ad68bca2d0f4c19b79b53d2e687df13c2083495a05b3f8e7e6ac
check_preimage_or_payload train_gaussian.py \
  ec7bbcd5f68341c2af4eec815ede7adeee186c7a87fe1ffcc292c2a1dc83d1fa

targets=(
  models/__init__.py
  models/gaussian.py
  models/edsr_face_iafm_idm_arm.py
  train_gaussian.py
  configs/train/face/probe_face_gaussian_iafm_idm_arm_seed1.yaml
  validate_face_iafm_idm_arm_configs.py
  test_face_iafm_idm_arm.py
  smoke_test_face_iafm_dataparallel.py
  evaluate_face_iafm_idm_arm.py
  analyze_face_iafm_idm_arm_probe.py
  test_analyze_face_iafm_idm_arm_probe.py
  test_train_microbatch_accumulation.py
  run_face_iafm_idm_arm_seed1_probe.sh
  README_face_iafm_idm_arm_seed1_20260913.txt
  THIRD_PARTY_NOTICES_IAFM_IDM_ARM.md
  建议E_IAFM_IDM_ARM迁移复核_20260913.md
)
for target in "${targets[@]}"; do
  [[ -f "$payload/$target" ]] || {
    echo "STOP: payload file is missing: $target" >&2
    exit 1
  }
  case "$target" in
    models/__init__.py|models/gaussian.py|train_gaussian.py)
      ;;
    *)
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
      ;;
  esac
done
for path in "$stamp" "$log" "$pid_file" \
    results/face_iafm_idm_arm_seed1_probe \
    save/face_gaussian_iafm_idm_arm_seed1_probe5 \
    results/face_gaussian_iafm_idm_arm_seed1_probe5; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

backup="server_backups/pre_iafm_idm_arm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/models"
cp -a models/__init__.py models/gaussian.py "$backup/models/"
cp -a train_gaussian.py "$backup/"
for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")

python -c 'import torch, yaml, tqdm'
python -m py_compile \
  models/__init__.py models/gaussian.py models/edsr_face_iafm_idm_arm.py \
  train_gaussian.py validate_face_iafm_idm_arm_configs.py \
  test_face_iafm_idm_arm.py smoke_test_face_iafm_dataparallel.py \
  evaluate_face_iafm_idm_arm.py analyze_face_iafm_idm_arm_probe.py \
  test_analyze_face_iafm_idm_arm_probe.py \
  test_train_microbatch_accumulation.py
bash -n run_face_iafm_idm_arm_seed1_probe.sh
python validate_face_iafm_idm_arm_configs.py \
  --baseline configs/train/face/probe_face_gaussian_baseline_seed1.yaml \
  --candidate configs/train/face/probe_face_gaussian_iafm_idm_arm_seed1.yaml \
  --expected-seed 1
python test_train_microbatch_accumulation.py
python test_analyze_face_iafm_idm_arm_probe.py
python test_face_iafm_idm_arm.py --device cpu
CUDA_VISIBLE_DEVICES="$admission_gpu" python test_face_iafm_idm_arm.py \
  --device cuda --gpu 0
CUDA_VISIBLE_DEVICES="$physical_gpus" python smoke_test_face_iafm_dataparallel.py \
  --physical-gpus "$physical_gpus"
{
  echo "IMPLEMENTATION_COMMIT=$implementation_commit"
  echo "PHYSICAL_GPUS=$physical_gpus"
  echo "ADMISSION_GPU=$admission_gpu"
  echo 'IAFM IDM+ARM CUDA ADMISSION PASSED'
} | tee "$stamp"

mkdir -p results
nohup env PHYSICAL_GPUS="$physical_gpus" \
  EVALUATION_GPU="$evaluation_gpu" \
  bash run_face_iafm_idm_arm_seed1_probe.sh \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: IAFM IDM+ARM runner exited during launch' >&2
  tail -n 100 "$log" >&2 || true
  exit 1
fi

echo "IAFM IDM+ARM SEED1 PROBE LAUNCHED: pid=$pid"
echo "IMPLEMENTATION_COMMIT=$implementation_commit"
echo "PHYSICAL_GPUS=$physical_gpus"
echo "EVALUATION_GPU=$evaluation_gpu"
echo "LOG=$log"
echo "BACKUP=$backup"
