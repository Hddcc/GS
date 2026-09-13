GaussianSR IAFM IDM+ARM seed-1 compatibility bundle v2 (2026-09-13)

Purpose
-------
This is Suggestion E: a clean-room, soft-only adaptation of the CVPR 2026
IAFMNet information density model (IDM) and affine recalibration module (ARM).
It preserves the EDSR backbone, Gaussian renderer, decoder, data, L1 loss,
optimizer, and five-epoch budget. It does not include IGRA sparse convolution
or replace the full network.

Only seed 1 is included. Do not run seed 2/3 unless seed 1 passes.

Upload this single file
-----------------------
face_iafm_idm_arm_seed1_bundle_v2_20260913.zip

Run on the server
-----------------
cd /root/userfolder_new/20260527GaussiSR/GaussianSR-main
conda activate GS
unzip -q face_iafm_idm_arm_seed1_bundle_v2_20260913.zip
PHYSICAL_GPUS=0,2,4 ADMISSION_GPU=4 EVALUATION_GPU=4 \
  bash face_iafm_idm_arm_seed1_bundle_v2_20260913/install_and_launch_face_iafm_idm_arm_seed1_v2_20260913.sh

The entrypoint verifies both SHA-256 manifests and the failed/completed
blind-SPD oracle terminal state. It preserves the diagnosed cumulative model
registry and appends only the IAFM registration. It accepts only the exact
known server files or an exact partial installation of this bundle, backs up
the three changed source files, runs CPU, single-GPU CUDA, and three-GPU
DataParallel admission, then launches training in the background.

Progress
--------
tail -n 100 results/face_iafm_idm_arm_seed1_probe_orchestrator.log

Training details are written to:
results/face_gaussian_iafm_idm_arm_seed1_probe5/train_console.log

Decision rule
-------------
The paired 100-image x2/x4/x8 held-out mean must be strictly positive, every
scale must be at least -0.030 dB, and train mean/final must be at least
-0.010/-0.020 dB. Entropy, IDM spatial variation, every ARM response, and the
candidate output response must also remain active. Failure stops this route;
passing only admits a later seed-2/3 package.

Published SR evidence reduces selection risk but does not guarantee a gain on
the GaussianSR CelebA setting. See the included migration review and
THIRD_PARTY_NOTICES_IAFM_IDM_ARM.md for evidence and provenance boundaries.
