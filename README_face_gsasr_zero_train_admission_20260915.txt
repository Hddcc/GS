GSASR paper EDSR zero-train face admission (2026-09-15)

Purpose
-------
This is an evaluation-only cross-domain admission for Suggestion F. It
compares the official GSASR paper-version EDSR/DIV2K checkpoints with the
existing GaussianSR face baseline on the same 100 CelebA held-out images at
x2, x4, and x8. It performs no training, fine-tuning, TTA, or post-processing.
It is not a formal paper result.

Single-file transfer and launch
-------------------------------
Upload only:

  face_gsasr_zero_train_admission_bundle_v3_20260915.zip

Then run from the GaussianSR project root in the GS conda environment:

  unzip -q face_gsasr_zero_train_admission_bundle_v3_20260915.zip
  PHYSICAL_GPUS=0,1 ADMISSION_GPU=1 bash face_gsasr_zero_train_admission_bundle_v3_20260915/install_and_launch_face_gsasr_zero_train_admission_20260915.sh

The official encoder and decoder weights are included in the ZIP; the server
does not need network access. The installer does not create or modify the GS
conda environment and does not overwrite GaussianSR model/training files. It
compiles the bundled CUDA renderer locally under the isolated third_party
directory, runs source, checkpoint, analyzer, and CUDA smoke tests, and then
launches the admission in the background.

Monitor
-------
  tail -f results/face_gsasr_zero_train_admission_orchestrator.log

The orchestrator streams all three scale progresses with x2/x4/x8 prefixes.
Per-scale copies remain available under:

  results/face_gsasr_zero_train_admission/candidate_x2.log
  results/face_gsasr_zero_train_admission/candidate_x4.log
  results/face_gsasr_zero_train_admission/candidate_x8.log

With two cards, x2 and x4 first run in parallel on physical GPUs 0 and 1;
x8 then runs on GPU 0. The GaussianSR baseline manifest is produced first on
GPU 1. Three-card mode remains supported when three ids are supplied. Both
systems receive LR tensors produced by the same pinned torchvision/PIL
bicubic path; source-image and LR byte hashes are paired and verified.

Decision rule
-------------
The mean paired RGB-PSNR delta over x2/x4/x8 must be strictly positive, every
scale delta must be at least -0.030 dB, and repeat-inference maximum error must
not exceed 1e-6. Passing only admits a later full migration/fine-tuning step.
Failure stops this direction without parameter scanning.

Compatibility and provenance
----------------------------
The bundle uses the official GSASR paper-version code at commit
9d2eb64a51303ce7a22fd672197185488b38e10a and preserves its Apache-2.0
license. Local compatibility edits remove the einops dependency, replace the
same tensor rearrangements with reshape/permute operations, and avoid one
NumPy-to-Torch ABI conversion. Server compatibility with its Python/PyTorch/
CUDA toolchain is intentionally decided by the install-time compilation and
CUDA smoke test, not assumed in advance. See THIRD_PARTY_NOTICES_GSASR.md.
