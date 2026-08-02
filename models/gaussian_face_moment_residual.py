import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import GaussianSplatter
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-moment-residual')
class FaceMomentResidualGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """Condition query residuals on signed local Gaussian moments."""

    moment_dim = 5

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
            residual_gate_init=0.50,
            moment_scale=2.0):
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
            residual_feature_dim=residual_feature_dim,
            residual_hidden_dim=residual_hidden_dim,
            scale_min=scale_min,
            scale_max=scale_max,
            max_rgb_residual=max_rgb_residual,
            residual_gate_init=residual_gate_init,
        )
        if moment_scale <= 0:
            raise ValueError('moment_scale must be positive.')
        self.moment_scale = moment_scale

        old_layers = list(self.residual_head.children())
        old_first = old_layers[0]
        expanded_first = nn.Linear(
            old_first.in_features + self.moment_dim,
            old_first.out_features,
        )
        with torch.no_grad():
            expanded_first.weight[:, :old_first.in_features].copy_(
                old_first.weight
            )
            expanded_first.bias.copy_(old_first.bias)
            weight_error = (
                expanded_first.weight[:, :old_first.in_features]
                - old_first.weight
            ).abs().max().item()
            bias_error = (
                expanded_first.bias - old_first.bias
            ).abs().max().item()
        self.base_residual_input_dim = old_first.in_features
        self.shared_initialization_error = max(weight_error, bias_error)
        self.residual_head = nn.Sequential(expanded_first, *old_layers[1:])
        self.last_moment_basis = None

    def local_moment_basis(self, relative_coord):
        u = self.moment_scale * relative_coord[:, :, 0]
        v = self.moment_scale * relative_coord[:, :, 1]
        envelope = torch.exp(-0.5 * (u.square() + v.square()))
        return torch.stack((
            u * envelope,
            v * envelope,
            (u.square() - 1) * envelope,
            u * v * envelope,
            (v.square() - 1) * envelope,
        ), dim=-1)

    def query_rgb(self, coord, scale, cell=None):
        base_prediction = GaussianSplatter.query_rgb(
            self,
            coord,
            scale,
            cell,
        )
        if cell is None:
            raise ValueError('cell is required for moment residual prediction.')

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
        moment_basis = self.local_moment_basis(relative_coord)
        residual_input = torch.cat((
            query_feature,
            relative_coord,
            relative_cell,
            scale_feature,
            moment_basis,
        ), dim=-1)
        residual = self.residual_head(residual_input)
        residual = (
            self.max_rgb_residual
            * torch.sigmoid(self.residual_gate_logit)
            * torch.tanh(residual)
        )
        self.last_moment_basis = moment_basis.detach()
        self.last_rgb_residual = residual.detach()
        return base_prediction + residual
