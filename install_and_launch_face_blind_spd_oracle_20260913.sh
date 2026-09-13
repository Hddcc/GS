#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
payload="$bundle_dir/payload"
implementation_commit=4466f93725dc6846edc3797a577488f0c5baf1c9
physical_gpu="${PHYSICAL_GPU:-4}"
stamp="$project/results/face_blind_spd_oracle_cuda_admission_20260913.txt"
log="$project/results/face_blind_spd_oracle_admission_orchestrator.log"
pid_file="$project/results/face_blind_spd_oracle_admission_orchestrator.pid"

[[ "${CONDA_DEFAULT_ENV:-}" == "GS" ]] || {
  echo 'STOP: activate the GS conda environment first' >&2
  exit 1
}
[[ "$physical_gpu" =~ ^(0|2|4)$ ]] || {
  echo 'STOP: PHYSICAL_GPU must be one of the available cards: 0, 2, 4' >&2
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
previous_log=results/face_multi_primitive_m4_seed1_probe_orchestrator.log
[[ -f "$previous_log" ]] && grep -qx \
  'MULTI-PRIMITIVE SEED1 PROBE COMPLETED' "$previous_log" || {
  echo 'STOP: multi-primitive probe has not reached a terminal state' >&2
  exit 1
}
for required in \
    save/face_gaussian_baseline_seed1_probe_full5/epoch-5.pth \
    configs/train/face/probe_face_gaussian_baseline_seed1.yaml; do
  [[ -f "$required" ]] || {
    echo "STOP: missing baseline artifact: $required" >&2
    exit 1
  }
done
celeba_count=$(find Dataset/FACE/CelebA/val/HR -maxdepth 1 -type f | wc -l)
helen_count=$(find Dataset/FACE/Helen/test/HR -maxdepth 1 -type f | wc -l)
[[ "$celeba_count" -ge 118 ]] || {
  echo "STOP: CelebA val requires at least 118 files; found $celeba_count" >&2
  exit 1
}
[[ "$helen_count" -ge 50 ]] || {
  echo "STOP: Helen test requires at least 50 files; found $helen_count" >&2
  exit 1
}

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

original_gaussian=e894e96e404fef42b252755f0b63abffee454f0ff13677646c0dac9a6adfc8f3
installed_gaussian=$(sha256sum models/gaussian.py | cut -d ' ' -f1)
payload_gaussian=$(sha256sum "$payload/models/gaussian.py" | cut -d ' ' -f1)
if [[ "$installed_gaussian" != "$original_gaussian" \
      && "$installed_gaussian" != "$payload_gaussian" ]]; then
  echo 'STOP: hash mismatch: models/gaussian.py' >&2
  echo "expected-original=$original_gaussian" >&2
  echo "expected-payload=$payload_gaussian" >&2
  echo "actual=$installed_gaussian" >&2
  exit 1
fi
check_hash models/edsr.py \
  47a3288dbf94d481e10ab1370505d0729dbbe3b382de33f4c8acdb7c4e907d0b

targets=(
  models/gaussian.py
  blind_spd_oracle.py
  evaluate_face_blind_spd_oracle.py
  analyze_face_blind_spd_oracle.py
  test_blind_spd_oracle_math.py
  smoke_test_face_blind_spd_oracle.py
  test_analyze_face_blind_spd_oracle.py
  configs/eval/face_blind_spd_oracle_protocol_20260913.json
  run_face_blind_spd_oracle_admission.sh
  README_face_blind_spd_oracle_20260913.txt
  THIRD_PARTY_NOTICES_BLIND_SPD_ORACLE.md
  建议D_盲人脸SPD退化补偿_研究复核_20260913.md
)
for target in "${targets[@]}"; do
  [[ -f "$payload/$target" ]] || {
    echo "STOP: payload file is missing: $target" >&2
    exit 1
  }
  if [[ "$target" != models/gaussian.py && -e "$target" ]]; then
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
    results/face_blind_spd_oracle_admission; do
  [[ ! -e "$path" ]] || {
    echo "STOP: output already exists; do not overwrite: $path" >&2
    exit 1
  }
done

backup="server_backups/pre_blind_spd_oracle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup/models"
cp -a models/gaussian.py "$backup/models/"
for target in "${targets[@]}"; do
  mkdir -p "$(dirname "$target")"
  cp -a "$payload/$target" "$target"
done
(cd "$project" && sha256sum -c "$bundle_dir/INSTALLED_SHA256")

python -c 'import torch, yaml'
python -m py_compile \
  models/gaussian.py blind_spd_oracle.py \
  evaluate_face_blind_spd_oracle.py analyze_face_blind_spd_oracle.py \
  test_blind_spd_oracle_math.py smoke_test_face_blind_spd_oracle.py \
  test_analyze_face_blind_spd_oracle.py
bash -n run_face_blind_spd_oracle_admission.sh
python test_blind_spd_oracle_math.py
python test_analyze_face_blind_spd_oracle.py
python smoke_test_face_blind_spd_oracle.py --device cpu
CUDA_VISIBLE_DEVICES="$physical_gpu" python smoke_test_face_blind_spd_oracle.py \
  --device cuda --gpu 0
{
  echo "IMPLEMENTATION_COMMIT=$implementation_commit"
  echo "PHYSICAL_GPU=$physical_gpu"
  echo 'BLIND-SPD ORACLE CUDA ADMISSION PASSED'
} | tee "$stamp"

mkdir -p results
nohup env PHYSICAL_GPU="$physical_gpu" \
  bash run_face_blind_spd_oracle_admission.sh \
  > "$log" 2>&1 &
pid=$!
echo "$pid" | tee "$pid_file"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo 'STOP: blind-SPD oracle runner exited during launch' >&2
  tail -n 100 "$log" >&2 || true
  exit 1
fi

echo "BLIND-SPD ORACLE LAUNCHED: pid=$pid"
echo "IMPLEMENTATION_COMMIT=$implementation_commit"
echo "PHYSICAL_GPU=$physical_gpu"
echo "LOG=$log"
echo "BACKUP=$backup"
