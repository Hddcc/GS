# Third-Party Research Notice

The multi-primitive experiment is informed by:

- `GSASR: Generalized and Efficient 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution`, ICCV 2025.
- Official repository commit `9d2eb64a51303ce7a22fd672197185488b38e10a`, Apache License 2.0.

This repository adaptation is independently implemented against GaussianSR's
existing Python interfaces. It does not copy GSASR's Transformer architecture,
Gaussian embedding implementation, or CUDA rasterizer. The uniform `m=4`
density is used as published experimental motivation, not claimed as a novel
contribution.
