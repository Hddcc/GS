from argparse import Namespace

import torch

from models import register
from models.edsr_geometry_appearance import GeometryAppearanceEncoder


def sparsemax(logits, dim=-1):
    """Project logits onto the probability simplex with sparse support."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = shifted.sort(dim=dim, descending=True).values
    cumulative = sorted_logits.cumsum(dim)
    support_range = torch.arange(
        1,
        logits.shape[dim] + 1,
        device=logits.device,
        dtype=logits.dtype,
    )
    view_shape = [1] * logits.ndim
    view_shape[dim] = -1
    support_range = support_range.view(view_shape)
    support = 1 + support_range * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(dim, support_size - 1) - 1
    ) / support_size.to(logits.dtype)
    return (shifted - threshold).clamp_min(0)


class SparseGaussianAllocationEncoder(GeometryAppearanceEncoder):
    """Use one sparse allocation rule for training and evaluation."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            fusion_dim=64,
            appearance_gate_init=0.25,
            allocation_temperature=1.0):
        super().__init__(
            args=args,
            n_class=n_class,
            shallow_depth=shallow_depth,
            middle_depth=middle_depth,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            appearance_gate_init=appearance_gate_init,
        )
        if allocation_temperature <= 0:
            raise ValueError('allocation_temperature must be positive.')
        self.allocation_temperature = allocation_temperature
        self.last_sparse_support = None

    def geometry_probabilities(self, logits):
        batch, classes, height, width = logits.shape
        flat = logits.permute(0, 2, 3, 1).reshape(-1, classes)
        probabilities = sparsemax(
            flat / self.allocation_temperature, dim=-1
        )
        self.last_sparse_support = (
            probabilities > 0
        ).sum(dim=-1).reshape(batch, height, width).detach()
        return probabilities.reshape(
            batch, height, width, classes
        ).permute(0, 3, 1, 2).contiguous()


@register('edsr-face-sparse-gaussian-allocation')
def make_sparse_gaussian_allocation_encoder(
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        scale=2,
        no_upsampling=True,
        rgb_range=1,
        n_class=100,
        use_pretrained=False,
        shallow_depth=4,
        middle_depth=10,
        geometry_dim=32,
        fusion_dim=64,
        appearance_gate_init=0.25,
        allocation_temperature=1.0):
    if use_pretrained:
        raise ValueError(
            'Sparse Gaussian allocation requires random initialization.'
        )
    args = Namespace(
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        res_scale=res_scale,
        scale=[scale],
        no_upsampling=no_upsampling,
        rgb_range=rgb_range,
        n_colors=3,
        pretrained_path=None,
    )
    return SparseGaussianAllocationEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        fusion_dim=fusion_dim,
        appearance_gate_init=appearance_gate_init,
        allocation_temperature=allocation_temperature,
    )
