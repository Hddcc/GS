GaussianSR four-primitive m=4 seed-1 bundle (2026-09-09)

Purpose
-------
This is operation recommendation C: reuse the admitted memory-efficient
renderer and replace one primitive per LR position with four independent
subpixel primitives. It is an ICCV 2025 GSASR-informed performance adaptation,
not a claim that multi-Gaussian densification is novel.

Only seed 1 and m=4 are included. Do not run seed 2/3 unless seed 1 passes.

Upload this single file
-----------------------
GaussianSR_face_multi_primitive_m4_seed1_bundle_20260909.zip

Run on the server
-----------------
cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
conda activate GS
unzip -q GaussianSR_face_multi_primitive_m4_seed1_bundle_20260909.zip
PHYSICAL_GPUS=0,2,4 ADMISSION_GPU=4 EVALUATION_GPU=4 \
  bash face_multi_primitive_m4_seed1_bundle_20260909/install_and_launch_face_multi_primitive_m4_seed1_20260909.sh

The entrypoint verifies both manifests, the terminated MSConvStar probe, the
admitted memory-efficient renderer, and the baseline artifacts. It installs a
complete payload, runs CPU/CUDA admission, and launches training only if every
check passes.

Progress
--------
tail -n 80 results/face_multi_primitive_m4_seed1_probe_orchestrator.log

Training details are written to:
results/face_gaussian_multi_primitive_m4_seed1_probe5/train_console.log

The quality gate uses paired baseline/candidate PSNR on 100 held-out images at
x2, x4, and x8. The aggregate mean must be positive. This experiment can still
fail on CelebA even though GSASR reports a positive m=1 to m=4 ablation on
DIV2K; the published evidence reduces selection risk but is not a guarantee.
