#!/usr/bin/env bash
set -euo pipefail

project=/root/userfolder_new/20260527GaussiSR/GaussianSR-main
physical_gpu="${PHYSICAL_GPU:-4}"
run_name="${RUN_NAME:-face_gaussian_frequency_residual_scale_edge_seed1_probe5}"

cd "$project"
echo "RUNTIME: physical GPU $physical_gpu"
echo 'RUNTIME: scale-conditioned edge weight, seed 1, 5 epochs'
echo 'START: scale-conditioned edge probe'
CUDA_VISIBLE_DEVICES="$physical_gpu" python train_gaussian.py \
  --config configs/train/face/probe_face_gaussian_frequency_residual_scale_edge_seed1.yaml \
  --name "$run_name" \
  --gpu 0 \
  --seed 1
echo "DONE: $run_name"
