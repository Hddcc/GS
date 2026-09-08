GaussianSR operation recommendation 2, stage 1
================================================

Purpose
-------
Install a numerically equivalent primitive-chunk rasterizer and measure real
CUDA peak allocated memory before any channel-budget training is created.

The direct K=3 query-gather draft was intentionally withdrawn. Its fixed
Sigma/100 conversion is only approximately correct at x4, clamped neighbors
double-count border primitives, and direct query features bypass the existing
3x3 coef/freq convolutions. This package changes only memory scheduling.

Server command
--------------
Run from the GaussianSR project root with the GS conda environment active:

  unzip GaussianSR_memory_efficient_raster_operation2_bundle_20260908.zip
  PHYSICAL_GPU=4 bash memory_efficient_raster_operation2_20260908/install_face_memory_efficient_raster_operation2_20260908.sh

PHYSICAL_GPU may be 0, 2, or 4. Only one GPU is used. No training is launched.
The installer is resumable when already-copied payload files match exactly.

Expected terminal line
----------------------
  MEMORY-EFFICIENT RASTER CUDA ADMISSION PASSED

Send back the complete terminal output. Do not start channel-budget training
until the measured output error, memory reduction, and time ratio are reviewed.

Implementation commit
---------------------
da41c9e17e26d7a015de2df5af0efcfd38b4a0df
