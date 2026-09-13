GaussianSR face blind-SPD oracle admission, 2026-09-13

Purpose
-------
This is an evaluation-only falsification gate for Suggestion D. It does not
train a degradation encoder and does not claim a blind-SR result. It uses the
known synthetic blur covariance only to test whether renderer covariance
addition or PSD-safe subtraction has any credible upper-bound benefit.

One-command use after uploading the single ZIP
------------------------------------------------
Run in the GS conda environment from the GaussianSR project root:

  unzip face_blind_spd_oracle_bundle_20260913.zip
  PHYSICAL_GPU=4 bash face_blind_spd_oracle_bundle_20260913/install_and_launch_face_blind_spd_oracle_20260913.sh

The installer verifies both SHA-256 manifests, backs up models/gaussian.py,
runs CPU and CUDA admission tests, and launches the oracle on physical GPU 4.
Cards 0 or 2 may be selected through PHYSICAL_GPU. No multi-GPU training is
performed.

Monitor
-------
  tail -f results/face_blind_spd_oracle_admission_orchestrator.log

The first stage evaluates 12 fixed variants on 18 CelebA validation images.
If no variant passes, the run stops before the 100-image confirmation. If one
passes, the top eligible variant is evaluated on 100 disjoint CelebA images;
Helen 50-image confirmation runs only if CelebA passes.

Terminal lines
--------------
  BLIND-SPD ORACLE PROBE FAILED
  BLIND-SPD ORACLE PROBE COMPLETED

or

  BLIND-SPD ORACLE PROBE PASSED
  BLIND-SPD ORACLE PROBE COMPLETED

Failure means Suggestion D stops permanently. Passing only admits a later,
fair blind-degradation training experiment; it is not itself a blind result.
