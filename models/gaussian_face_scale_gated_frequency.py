import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import GaussianSplatter
from models.gaussian_face_frequency_residual import FaceFrequencyResidualGaussianSplatter


@register('gaussian-splatter-face-scale-gated-frequency')
class ScaleGatedFrequencyGaussianSplatter(FaceFrequencyResidualGaussianSplatter):
    """Zero-initialized directional residual with optional query-scale gating."""

    def __init__(self, *args, use_scale_gate=True, gate_hidden_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_scale_gate = use_scale_gate
        feature_dim = self.residual_head[0].in_features - 8
        if gate_hidden_dim <= 0:
            raise ValueError('gate_hidden_dim must be positive.')
        self.scale_gate = nn.Sequential(
            nn.Linear(feature_dim + 6, gate_hidden_dim),
            nn.SiLU(),
            nn.Linear(gate_hidden_dim, 1),
        ) if use_scale_gate else None
        # Only the final residual layer is zero: it receives gradients on step 1.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.residual_gate_logit.requires_grad_(False)
        self.last_scale_gate = None

    def adaptation_parameters(self):
        for module in (self.frequency_encoder, self.residual_head, self.scale_gate):
            if module is not None:
                yield from module.parameters()

    def query_rgb(self, coord, scale, cell=None):
        if cell is None:
            raise ValueError('cell is required for scale-gated residual prediction.')
        base = GaussianSplatter.query_rgb(self, coord, scale, cell)
        grid = coord.flip(-1).unsqueeze(1)
        feature = F.grid_sample(
            self.residual_feature, grid, mode='bilinear', align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        centers = F.grid_sample(
            self.feat_coord, grid, mode='nearest', align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        size = coord.new_tensor(self.feat.shape[-2:])
        relative = (coord - centers) * size
        relative_cell = cell * size
        scale_feature = self.scale_features(
            scale, *coord.shape[:2], feature.dtype, feature.device,
        )
        residual_input = torch.cat(
            (feature, relative, relative_cell, scale_feature), dim=-1,
        )
        if self.scale_gate is None:
            gate = torch.full_like(coord[:, :, :1], 0.5)
        else:
            gate = torch.sigmoid(self.scale_gate(torch.cat(
                (feature, relative_cell, scale_feature), dim=-1,
            )))
        residual = self.max_rgb_residual * gate * torch.tanh(
            self.residual_head(residual_input)
        )
        self.last_scale_gate = gate.detach()
        self.last_rgb_residual = residual.detach()
        return base + residual
