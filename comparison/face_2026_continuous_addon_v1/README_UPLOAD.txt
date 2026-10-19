FACE 2026 CONTINUOUS-SCALE ADDON (2026-10-19)

Put this directory inside face_comparison_bundle_v7. It uses the existing
datasets/face/CelebA/train/HR and val/HR PNG directories. No data or Markdown
files are included. DIIF files and processes are untouched.

FTD-LIIF: IEEE TMM 2026, CCF B, pinned upstream 82d659c824355f84f1d6bcf22d1df7cf8e8b3231.
FPLIA: ECCV 2026, CCF B, pinned upstream cd71335aa95a2d5129e47aa787f38bbfc5233b25.
The pinned source trees are included so no GitHub clone is needed on the
server. FTD-LIIF carries its BSD license. FPLIA has no license file upstream;
use the bundled copy for this private research run, not public redistribution.

From face_comparison_bundle_v7:

  GPU_IDS=0,1 EPOCH_MAX=1000 bash face_2026_continuous_addon_v1/run_face_2026.sh ftd train

  GPU_IDS=2 FPLIA_EPOCHS=1000 bash face_2026_continuous_addon_v1/run_face_2026.sh fplia train

After a training run finishes:

  GPU_IDS=0 bash face_2026_continuous_addon_v1/run_face_2026.sh ftd eval
  GPU_IDS=2 bash face_2026_continuous_addon_v1/run_face_2026.sh fplia eval

Do not launch both on a GPU already occupied by DIIF. FPLIA uses the first
GPU in GPU_IDS and creates a separate conda environment; FTD uses the GS env.
FPLIA downloads PyTorch CUDA 12.1 wheels. Network access and free disk space
are required. The original B200/PyTorch 2.8 training environment is not
available on the reported 3090/driver 530 server, so PyTorch 2.4.1 is an
adaptation. The generated VAE config uses SwinIR to match the preceding LIA
checkpoint and following diffusion stage (the upstream VAE example says RDN).
Run setup/smoke before committing to a full training schedule.

Logs:
  ftd_liif/results/train_console.log
  fplia/results/{lia-swinir,vae,diffusion,finetune}_train_console.log

Checkpoints:
  ftd_liif/source/save/face_ftd_liif/epoch-last.pth
  fplia/results/finetune_checkpoint.path (contains actual checkpoint path)

The six-scale evaluation adapters use one checkpoint and output face_scales.csv
and face_scales_per_image.csv. They have not been validated on the server.
Do not add these methods to the paper results table until the training and
evaluation runs complete and their outputs are checked.
