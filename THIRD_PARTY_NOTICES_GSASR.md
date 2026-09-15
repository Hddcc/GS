# Third-Party Notice: GSASR

This experiment redistributes a fixed subset of the official GSASR paper
implementation from commit `9d2eb64a51303ce7a22fd672197185488b38e10a`:

- Project: Generalized and Efficient 2D Gaussian Splatting for Arbitrary-scale
  Super-Resolution (ICCV 2025)
- Repository: https://github.com/ChrisDud0257/GSASR
- License: Apache License 2.0

The original Apache-2.0 `LICENSE` is retained at
`third_party/gsasr_paper_9d2eb64/LICENSE`. The bundled EDSR paper-version
weights are the authors' published `GSASR_paper/EDSR` checkpoints and their
SHA-256 digests are pinned by the admission analyzer and package manifests.

Local compatibility changes are intentionally limited to removing unused
imports, replacing `einops.rearrange` calls with equivalent PyTorch
reshape/permute operations, and constructing one position-index tensor
without `torch.from_numpy`. These changes do not alter trained parameters or
the intended model equations.
