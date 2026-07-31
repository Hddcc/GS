import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian_local_freq_scale import LocalFrequencyScaleGaussianSplatter


def zero_last_layer(module):
    nn.init.zeros_(module[-1].weight)
    nn.init.zeros_(module[-1].bias)


class FactorizedFrequencyScaleModulator(nn.Module):
    """Separate local direction evidence from smooth scale conditioning."""

    def __init__(self, hidden_dim=32, scale_min=1.5, scale_max=8.0):
        super().__init__()
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError('Expected 0 < scale_min < scale_max.')

        haar_filters = torch.tensor([
            [[1.0, -1.0], [1.0, -1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]).unsqueeze(1) / 2
        self.register_buffer('haar_filters', haar_filters)
        self.log_scale_min = math.log(scale_min)
        self.log_scale_range = math.log(scale_max) - self.log_scale_min

        self.frequency_encoder = nn.Sequential(
            nn.Conv2d(4, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.frequency_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1),
        )
        self.scale_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.scale_gain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )

        zero_last_layer(self.frequency_head)
        zero_last_layer(self.size_head)
        zero_last_layer(self.scale_gain_head)

    def directional_frequency(self, patch_features):
        batch, channels, height, width = patch_features.shape
        padded = F.pad(patch_features, (0, 1, 0, 1), mode='replicate')
        filters = self.haar_filters.repeat(channels, 1, 1, 1).to(
            dtype=patch_features.dtype
        )
        response = F.conv2d(padded, filters, groups=channels).reshape(
            batch, channels, 3, height, width
        )
        directional = torch.sqrt(response.square().mean(dim=1) + 1e-8)
        energy = torch.sqrt(directional.square().sum(dim=1, keepdim=True) + 1e-8)
        return directional, energy[:, 0]

    def scale_features(self, patch_scales, dtype):
        scales = patch_scales.to(dtype=dtype).clamp_min(1e-6)
        normalized = (
            (scales.log() - self.log_scale_min) / self.log_scale_range
        ).clamp(0, 1)
        return torch.stack((
            normalized,
            normalized.square(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ), dim=-1)

    def forward(self, patch_features, patch_scales):
        directional, energy = self.directional_frequency(patch_features)
        direction_ratio = directional / directional.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        mean_energy = energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        normalized_energy = torch.log1p(energy / mean_energy).unsqueeze(1)
        frequency_input = torch.cat((direction_ratio, normalized_energy), dim=1)
        frequency_embedding = self.frequency_encoder(frequency_input)
        frequency_raw = self.frequency_head(frequency_embedding)

        scale_embedding = self.scale_encoder(
            self.scale_features(patch_scales, patch_features.dtype)
        )
        size_raw = self.size_head(scale_embedding).unsqueeze(-1).unsqueeze(-1)
        scale_gains = 2 * torch.sigmoid(
            self.scale_gain_head(scale_embedding)
        ).unsqueeze(-1).unsqueeze(-1)
        frequency_raw = frequency_raw * scale_gains
        size_raw = size_raw.expand(
            -1, -1, patch_features.shape[-2], patch_features.shape[-1]
        )
        size_raw = size_raw + frequency_raw[:, :1]
        return torch.cat((size_raw, frequency_raw[:, 1:]), dim=1)


@register('gaussian-splatter-local-frequency-scale-v2')
class FactorizedLocalFrequencyScaleGaussianSplatter(
        LocalFrequencyScaleGaussianSplatter):
    """GaussianSR with factorized size, anisotropy, orientation and opacity."""

    def __init__(
            self,
            encoder_spec,
            dec_spec,
            kernel_size,
            hidden_dim=256,
            unfold_row=7,
            unfold_column=7,
            num_points=100,
            frequency_hidden_dim=32,
            scale_min=1.5,
            scale_max=8.0,
            max_log_size_delta=0.12,
            max_log_anisotropy_delta=0.10,
            max_rho_delta=0.08,
            max_opacity_logit_delta=0.25,
            residual_gate_init=0.25):
        if not 0 < residual_gate_init < 1:
            raise ValueError('residual_gate_init must be between 0 and 1.')
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
            frequency_hidden_dim=frequency_hidden_dim,
            max_log_sigma_delta=(
                max_log_size_delta + max_log_anisotropy_delta
            ),
            max_rho_delta=max_rho_delta,
            max_opacity_logit_delta=max_opacity_logit_delta,
        )
        self.local_modulator = FactorizedFrequencyScaleModulator(
            hidden_dim=frequency_hidden_dim,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        self.max_log_size_delta = max_log_size_delta
        self.max_log_anisotropy_delta = max_log_anisotropy_delta
        gate_logit = math.log(residual_gate_init / (1 - residual_gate_init))
        self.residual_gate_logits = nn.Parameter(torch.full((4,), gate_logit))
        self.last_gaussian_deltas = None

    def bounded_gaussian_deltas(self, raw_residuals):
        gates = torch.sigmoid(self.residual_gate_logits)
        size = (
            self.max_log_size_delta * gates[0]
            * F.softsign(raw_residuals[:, 0])
        )
        anisotropy = (
            self.max_log_anisotropy_delta * gates[1]
            * F.softsign(raw_residuals[:, 1])
        )
        return {
            'delta_sigma_x': size + anisotropy,
            'delta_sigma_y': size - anisotropy,
            'delta_rho': (
                self.max_rho_delta * gates[2]
                * F.softsign(raw_residuals[:, 2])
            ),
            'delta_opacity': (
                self.max_opacity_logit_delta * gates[3]
                * F.softsign(raw_residuals[:, 3])
            ),
        }

    def residual_saturation_ratios(self, raw_residuals):
        size_saturated = F.softsign(raw_residuals[:, 0]).abs() > 0.95
        anisotropy_saturated = F.softsign(raw_residuals[:, 1]).abs() > 0.95
        sigma_saturated = torch.logical_or(
            size_saturated, anisotropy_saturated
        ).float().mean().item()
        return {
            'delta_sigma_x': sigma_saturated,
            'delta_sigma_y': sigma_saturated,
            'delta_rho': float(
                (F.softsign(raw_residuals[:, 2]).abs() > 0.95)
                .float().mean().item()
            ),
            'delta_opacity': float(
                (F.softsign(raw_residuals[:, 3]).abs() > 0.95)
                .float().mean().item()
            ),
        }

    def local_gaussian_parameters(self, logits, patch_features, patch_scales):
        probabilities = logits.permute(0, 2, 3, 1)
        sigma_x = (
            probabilities * self.sigma_x.view(1, 1, 1, -1)
        ).sum(dim=-1)
        sigma_y = (
            probabilities * self.sigma_y.view(1, 1, 1, -1)
        ).sum(dim=-1)
        opacity = (
            probabilities * self.opacity[:, 0].view(1, 1, 1, -1)
        ).sum(dim=-1)
        rho = (
            probabilities * self.rho[:, 0].view(1, 1, 1, -1)
        ).sum(dim=-1)

        raw_residuals = self.local_modulator(patch_features, patch_scales)
        deltas = self.bounded_gaussian_deltas(raw_residuals)
        sigma_x = sigma_x.clamp_min(0.05) * torch.exp(
            deltas['delta_sigma_x']
        )
        sigma_y = sigma_y.clamp_min(0.05) * torch.exp(
            deltas['delta_sigma_y']
        )
        rho = (rho + deltas['delta_rho']).clamp(-0.95, 0.95)
        opacity = opacity.clamp(1e-4, 1 - 1e-4)
        opacity = torch.sigmoid(
            torch.logit(opacity) + deltas['delta_opacity']
        )

        patch_count = logits.shape[0]
        sigma_x = sigma_x.reshape(patch_count, -1)
        sigma_y = sigma_y.reshape(patch_count, -1)
        rho = rho.reshape(patch_count, -1)
        opacity = opacity.reshape(patch_count, -1)
        self.last_gaussian_deltas = {
            name: value.detach().reshape(patch_count, -1)
            for name, value in deltas.items()
        }
        self.last_gaussian_parameters = tuple(
            value.detach() for value in (sigma_x, sigma_y, rho, opacity)
        )
        return sigma_x, sigma_y, opacity, rho
