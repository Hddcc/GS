GaussianSR operation 2: Gaussian channel-budget seed-1 screen
Date: 2026-09-08
Implementation commit: 004d2d3260e308dfda82441d85805df2e8594ff8

Purpose
-------
The operation-2 CUDA admission passed with zero output error and a 52.14%
peak allocated-memory reduction. This bundle now compares Cg=16/32/64 against
the existing mathematically equivalent Cg=8 seed-1 baseline. It changes only
the Gaussian/bicubic 64-channel split and does not change parameter count.

The three candidates run concurrently:
  Cg=16 -> physical GPU 0
  Cg=32 -> physical GPU 2
  Cg=64 -> physical GPU 4

Each process uses microbatch 4 x 3 for effective batch 12. The run stops after
the seed-1 ranking. Do not launch seed 2/3 until the analysis is returned.

One-file transfer and launch
----------------------------
Upload only:
  GaussianSR_face_gaussian_channel_budget_seed1_bundle_20260908.zip

Then run:
  conda activate GS
  cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
  sha256sum GaussianSR_face_gaussian_channel_budget_seed1_bundle_20260908.zip
  unzip -o GaussianSR_face_gaussian_channel_budget_seed1_bundle_20260908.zip
  PHYSICAL_GPUS=0,2,4 bash \
    face_gaussian_channel_budget_seed1_bundle_20260908/install_and_launch_face_gaussian_channel_budget_seed1_20260908.sh

Monitor
-------
  tail -n 60 results/face_gaussian_channel_budget_seed1_screen_orchestrator.log

Per-candidate live logs:
  tail -f results/face_gaussian_channel_budget_seed1_screen/worker_cg16.log
  tail -f results/face_gaussian_channel_budget_seed1_screen/worker_cg32.log
  tail -f results/face_gaussian_channel_budget_seed1_screen/worker_cg64.log

Return after completion
-----------------------
Return the final orchestrator log containing the Cg table, ranking,
SELECTED_GAUSSIAN_CHANNELS line, and seed-1 screen status.
