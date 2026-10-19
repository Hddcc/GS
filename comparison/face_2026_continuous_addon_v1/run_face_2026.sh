#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:?usage: bash run_face_2026.sh ftd|fplia [setup|train]}"
ACTION="${2:-train}"
DATA_ROOT="${DATA_ROOT:-$ROOT/../datasets}"
TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT/face/CelebA/train/HR}"
VAL_ROOT="${VAL_ROOT:-$DATA_ROOT/face/CelebA/val/HR}"
GPU_IDS="${GPU_IDS:-0}"
TEST_ROOT="${TEST_ROOT:-$DATA_ROOT/face/CelebA/test/HR}"
SCALES="${SCALES:-2 2.5 3.5 4 5.5 8}"
SEED="${SEED:-1}"
EPOCH_MAX="${EPOCH_MAX:-1000}"
FPLIA_ENV="${FPLIA_ENV:-fplia_face_2026}"
FTD_ENV="${FTD_ENV:-GS}"
EVAL_EXTRA=()
[[ -z "${FIRST_K:-}" ]] || EVAL_EXTRA=(--first-k "$FIRST_K")

require_data() {
  for dir in "$TRAIN_ROOT" "$VAL_ROOT"; do
    [[ -d "$dir" ]] || { echo "Missing face data: $dir" >&2; exit 2; }
    compgen -G "$dir/*.png" >/dev/null || { echo "No PNG images: $dir" >&2; exit 2; }
  done
}

setup_ftd() {
  conda run --no-capture-output -n "$FTD_ENV" python -m pip install \
    'tensorboardX>=2.6,<3' 'piq>=0.8,<1' pyyaml tqdm imageio scikit-image
  conda run --no-capture-output -n "$FTD_ENV" python -c \
    'import torch, torchvision, yaml, tensorboardX, piq; assert torch.cuda.is_available()'
}

setup_fplia() {
  local source="$ROOT/fplia/source"
  if [[ ! -f "$source/main.py" || ! -f "$source/diffusion/models/finetune.py" ]]; then
    echo "Incomplete bundled FPLIA source: $source" >&2
    exit 2
  fi
  grep -Fxq cd71335aa95a2d5129e47aa787f38bbfc5233b25 \
    "$ROOT/fplia/SOURCE_COMMIT.txt" || { echo "FPLIA source marker mismatch" >&2; exit 2; }

  if ! conda env list | awk '{print $1}' | grep -Fxq "$FPLIA_ENV"; then
    conda create -y -n "$FPLIA_ENV" python=3.10
  fi
  conda run --no-capture-output -n "$FPLIA_ENV" python -m pip install 'pip<24.1'
  conda run --no-capture-output -n "$FPLIA_ENV" python -m pip install \
    torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  conda run --no-capture-output -n "$FPLIA_ENV" python -m pip install \
    'numpy<2' 'pytorch-lightning==1.8.4' 'torchmetrics==0.11.4' \
    einops==0.8.1 lpips==0.1.4 omegaconf==2.3.0 \
    taming-transformers-rom1504==0.0.6 timm imageio tensorboard pyyaml scikit-image
  CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" conda run --no-capture-output -n "$FPLIA_ENV" \
    python -c 'import torch, pytorch_lightning, timm, taming; assert torch.cuda.is_available()'
}

case "$METHOD:$ACTION" in
  ftd:setup) setup_ftd ;;
  fplia:setup) setup_fplia ;;
  ftd:train)
    require_data
    setup_ftd
    resume_args=()
    ftd_checkpoint="$ROOT/ftd_liif/source/save/face_ftd_liif/epoch-last.pth"
    if [[ "${RESUME:-1}" == 1 && -f "$ftd_checkpoint" ]]; then
      resume_args=(--resume "$ftd_checkpoint")
    fi
    conda run --no-capture-output -n "$FTD_ENV" python "$ROOT/prepare_configs.py" ftd \
      --addon "$ROOT" --train "$TRAIN_ROOT" --val "$VAL_ROOT" \
      --batch-size "${BATCH_SIZE:-4}" --epochs "$EPOCH_MAX" "${resume_args[@]}"
    mkdir -p "$ROOT/ftd_liif/results"
    cd "$ROOT/ftd_liif/source"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" conda run --no-capture-output -n "$FTD_ENV" \
      python train_liif_lrtfr.py --config "$ROOT/ftd_liif/config/face_ftd_liif.yaml" \
      --name face_ftd_liif --gpu "$GPU_IDS" --seed "$SEED" \
      2>&1 | tee "$ROOT/ftd_liif/results/train_console.log"
    ;;
  fplia:train)
    require_data
    setup_fplia
    mkdir -p "$ROOT/fplia/results" "$ROOT/fplia/logs"
    cd "$ROOT/fplia/source"
    lia_ckpt=""
    vae_ckpt=""
    diffusion_ckpt=""
    for stage in lia-swinir vae diffusion finetune; do
      marker="$ROOT/fplia/results/${stage}_checkpoint.path"
      if [[ -f "$marker" ]]; then
        checkpoint="$(<"$marker")"
        [[ -f "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
      else
        args=(fplia --stage "$stage" --addon "$ROOT" --train "$TRAIN_ROOT" \
          --val "$VAL_ROOT" --batch-size "${FPLIA_BATCH_SIZE:-1}" \
          --epochs "${FPLIA_EPOCHS:-$EPOCH_MAX}")
        [[ -z "$lia_ckpt" ]] || args+=(--lia-checkpoint "$lia_ckpt")
        [[ -z "$vae_ckpt" ]] || args+=(--vae-checkpoint "$vae_ckpt")
        [[ -z "$diffusion_ckpt" ]] || args+=(--diffusion-checkpoint "$diffusion_ckpt")
        conda run --no-capture-output -n "$FPLIA_ENV" python "$ROOT/prepare_configs.py" "${args[@]}"
        resume_args=(--name "face_${stage}")
        prior_run="$(find "$ROOT/fplia/logs" -mindepth 1 -maxdepth 1 -type d \
          -name "*_face_${stage}" | sort | tail -n 1)"
        if [[ "${RESUME:-1}" == 1 && -f "$prior_run/checkpoints/last.ckpt" ]]; then
          resume_args=(--resume "$prior_run")
        fi
        CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" conda run --no-capture-output -n "$FPLIA_ENV" \
          python main.py --base "$ROOT/fplia/config/face_${stage}.yaml" \
          "${resume_args[@]}" --logdir "$ROOT/fplia/logs" --seed "$SEED" \
          2>&1 | tee "$ROOT/fplia/results/${stage}_train_console.log"
        checkpoint="$(find "$ROOT/fplia/logs" -path "*/checkpoints/last.ckpt" \
          -path "*_face_${stage}/checkpoints/last.ckpt" -type f | sort | tail -n 1)"
        [[ -n "$checkpoint" && -f "$checkpoint" ]] || {
          echo "Training ended without last.ckpt for $stage" >&2; exit 2;
        }
        printf '%s\n' "$checkpoint" > "$marker"
      fi
      case "$stage" in
        lia-swinir) lia_ckpt="$checkpoint" ;;
        vae) vae_ckpt="$checkpoint" ;;
        diffusion) diffusion_ckpt="$checkpoint" ;;
      esac
      echo "DONE $stage: $checkpoint"
    done
    ;;
  ftd:eval)
    setup_ftd
    checkpoint="${CHECKPOINT:-$ROOT/ftd_liif/source/save/face_ftd_liif/epoch-last.pth}"
    [[ -f "$checkpoint" ]] || { echo "Missing FTD checkpoint: $checkpoint" >&2; exit 2; }
    [[ -d "$TEST_ROOT" ]] || { echo "Missing test data: $TEST_ROOT" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" conda run --no-capture-output -n "$FTD_ENV" \
      python "$ROOT/evaluate_ftd.py" --source "$ROOT/ftd_liif/source" \
      --checkpoint "$checkpoint" --config "$ROOT/ftd_liif/config/face_ftd_liif.yaml" \
      --images "$TEST_ROOT" --scales $SCALES \
      --output "$ROOT/ftd_liif/results/face_scales.csv" \
      "${EVAL_EXTRA[@]}"
    ;;
  fplia:eval)
    [[ -d "$TEST_ROOT" ]] || { echo "Missing test data: $TEST_ROOT" >&2; exit 2; }
    checkpoint="${CHECKPOINT:-}"
    if [[ -z "$checkpoint" ]]; then
      marker="$ROOT/fplia/results/finetune_checkpoint.path"
      [[ -f "$marker" ]] || { echo "Missing FPLIA checkpoint marker: $marker" >&2; exit 2; }
      checkpoint="$(<"$marker")"
    fi
    [[ -f "$checkpoint" ]] || { echo "Missing FPLIA checkpoint: $checkpoint" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" conda run --no-capture-output -n "$FPLIA_ENV" \
      python "$ROOT/evaluate_fplia.py" --source "$ROOT/fplia/source" \
      --checkpoint "$checkpoint" --config "$ROOT/fplia/config/face_finetune.yaml" \
      --images "$TEST_ROOT" --scales $SCALES --steps "${DDIM_STEPS:-200}" \
      --output "$ROOT/fplia/results/face_scales.csv" \
      "${EVAL_EXTRA[@]}"
    ;;
  *) echo "Unknown method/action: $METHOD $ACTION" >&2; exit 2 ;;
esac
