import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import GaussianSplatter


class DirectionalFrequencyEncoder(nn.Module):
    """Encode normalized directional evidence together with Gaussian features."""

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        filters = torch.tensor([
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            [[-2.0, -1.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 1.0, 2.0]],
            [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
        ]).unsqueeze(1) / 8
        self.register_buffer('directional_filters', filters)
        self.body = nn.Sequential(
            nn.Conv2d(feature_dim + 5, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def directional_features(self, inp):
        if inp.shape[1] == 3:
            luminance = (
                0.299 * inp[:, 0:1]
                + 0.587 * inp[:, 1:2]
                + 0.114 * inp[:, 2:3]
            )
        else:
            luminance = inp.mean(dim=1, keepdim=True)
        response = F.conv2d(
            F.pad(luminance, (1, 1, 1, 1), mode='replicate'),
            self.directional_filters.to(dtype=inp.dtype),
        ).abs()
        energy = torch.sqrt(response.square().sum(dim=1, keepdim=True) + 1e-8)
        direction_ratio = response / response.sum(dim=1, keepdim=True).clamp_min(1e-6)
        mean_energy = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized_energy = torch.log1p(energy / mean_energy)
        return torch.cat((direction_ratio, normalized_energy), dim=1)

    def forward(self, feature, inp):
        frequency = self.directional_features(inp)
        return self.body(torch.cat((feature, frequency), dim=1))


@register('gaussian-splatter-face-frequency-residual')
class FaceFrequencyResidualGaussianSplatter(GaussianSplatter):
    """GaussianSR with query-wise directional frequency residual correction."""

    def __init__(
            self,
            encoder_spec,
            dec_spec,
            kernel_size,
            hidden_dim=256,
            unfold_row=7,
            unfold_column=7,
            num_points=100,
            residual_feature_dim=48,
            residual_hidden_dim=64,
            scale_min=1.5,
            scale_max=8.0,
            max_rgb_residual=0.20,
            residual_gate_init=0.50):
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
        )
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError('Expected 0 < scale_min < scale_max.')
        if not 0 < residual_gate_init < 1:
            raise ValueError('residual_gate_init must be between 0 and 1.')
        if max_rgb_residual <= 0:
            raise ValueError('max_rgb_residual must be positive.')

        self.frequency_encoder = DirectionalFrequencyEncoder(
            self.encoder.out_dim, residual_feature_dim
        )
        query_dim = residual_feature_dim + 2 + 2 + 4
        self.residual_head = nn.Sequential(
            nn.Linear(query_dim, residual_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(residual_hidden_dim, residual_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(residual_hidden_dim, 3),
        )
        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_head[-1].bias)

        self.log_scale_min = math.log(scale_min)
        self.log_scale_range = math.log(scale_max) - self.log_scale_min
        self.max_rgb_residual = max_rgb_residual
        gate_logit = math.log(residual_gate_init / (1 - residual_gate_init))
        self.residual_gate_logit = nn.Parameter(torch.tensor(gate_logit))
        self.residual_feature = None
        self.last_rgb_residual = None

    def gen_feat(self, inp):
        feature, logits = super().gen_feat(inp)
        self.residual_feature = self.frequency_encoder(feature, inp)
        return feature, logits

    def scale_features(self, scale, batch_size, query_count, dtype, device):
        scale = scale.reshape(-1).to(device=device, dtype=dtype)
        if scale.numel() == 1:
            scale = scale.expand(batch_size)
        if scale.numel() != batch_size:
            raise ValueError('Expected one scale per batch sample.')
        normalized = (
            (scale.clamp_min(1e-6).log() - self.log_scale_min)
            / self.log_scale_range
        ).clamp(0, 1)
        features = torch.stack((
            normalized,
            normalized.square(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ), dim=-1)
        return features.unsqueeze(1).expand(-1, query_count, -1)

    def query_rgb(self, coord, scale, cell=None):
        base_prediction = super().query_rgb(coord, scale, cell)
        if cell is None:
            raise ValueError('cell is required for frequency residual prediction.')

        query_grid = coord.flip(-1).unsqueeze(1)
        query_feature = F.grid_sample(
            self.residual_feature,
            query_grid,
            mode='bilinear',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        query_feature_coord = F.grid_sample(
            self.feat_coord,
            query_grid,
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        relative_coord = coord - query_feature_coord
        relative_coord = relative_coord.clone()
        relative_coord[:, :, 0] *= self.feat.shape[-2]
        relative_coord[:, :, 1] *= self.feat.shape[-1]
        relative_cell = cell.clone()
        relative_cell[:, :, 0] *= self.feat.shape[-2]
        relative_cell[:, :, 1] *= self.feat.shape[-1]

        batch_size, query_count = coord.shape[:2]
        scale_feature = self.scale_features(
            scale,
            batch_size,
            query_count,
            query_feature.dtype,
            query_feature.device,
        )
        residual_input = torch.cat((
            query_feature,
            relative_coord,
            relative_cell,
            scale_feature,
        ), dim=-1)
        residual = self.residual_head(residual_input)
        residual = (
            self.max_rgb_residual
            * torch.sigmoid(self.residual_gate_logit)
            * torch.tanh(residual)
        )
        self.last_rgb_residual = residual.detach()
        return base_prediction + residual
