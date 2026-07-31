import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import (
    GaussianSplatter,
    fetching_features_from_tensor,
    generate_meshgrid,
)


class DirectionalFrequencyScaleModulator(nn.Module):
    """Predict bounded local Gaussian residuals from frequency and scale."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        haar_filters = torch.tensor([
            [[1.0, -1.0], [1.0, -1.0]],
            [[1.0, 1.0], [-1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]).unsqueeze(1) / 2
        self.register_buffer('haar_filters', haar_filters)

        self.scale_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(4 + hidden_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1),
        )

        # Start from local GaussianSR parameters and learn bounded residuals.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, patch_features, patch_scales):
        gray = patch_features.mean(dim=1, keepdim=True)
        gray = F.pad(gray, (0, 1, 0, 1), mode='replicate')
        directional = F.conv2d(
            gray,
            self.haar_filters.to(dtype=gray.dtype),
        ).abs()
        energy = torch.sqrt(directional.pow(2).sum(dim=1, keepdim=True) + 1e-8)
        frequency = torch.cat((directional, energy), dim=1)

        scales = patch_scales.to(dtype=patch_features.dtype).clamp_min(1e-6)
        scale_features = torch.stack((
            scales / 8.0,
            scales.reciprocal(),
            scales.log() / math.log(8.0),
        ), dim=-1)
        scale_embedding = self.scale_encoder(scale_features)
        scale_embedding = scale_embedding.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, patch_features.shape[-2], patch_features.shape[-1]
        )
        return self.fusion(torch.cat((frequency, scale_embedding), dim=1))


@register('gaussian-splatter-local-frequency-scale')
class LocalFrequencyScaleGaussianSplatter(GaussianSplatter):
    """GaussianSR with per-patch, frequency- and scale-conditioned geometry."""

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
            max_log_sigma_delta=0.35,
            max_rho_delta=0.25,
            max_opacity_logit_delta=1.0):
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
        )
        self.local_modulator = DirectionalFrequencyScaleModulator(
            hidden_dim=frequency_hidden_dim
        )
        self.max_log_sigma_delta = max_log_sigma_delta
        self.max_rho_delta = max_rho_delta
        self.max_opacity_logit_delta = max_opacity_logit_delta
        self.last_gaussian_parameters = None

    def local_gaussian_parameters(self, logits, patch_features, patch_scales):
        probabilities = logits.permute(0, 2, 3, 1)
        sigma_x_bank = self.sigma_x.view(1, 1, 1, -1)
        sigma_y_bank = self.sigma_y.view(1, 1, 1, -1)
        opacity_bank = self.opacity[:, 0].view(1, 1, 1, -1)
        rho_bank = self.rho[:, 0].view(1, 1, 1, -1)

        sigma_x = (probabilities * sigma_x_bank).sum(dim=-1)
        sigma_y = (probabilities * sigma_y_bank).sum(dim=-1)
        opacity = (probabilities * opacity_bank).sum(dim=-1)
        rho = (probabilities * rho_bank).sum(dim=-1)

        residuals = self.local_modulator(patch_features, patch_scales)
        delta_sigma_x = self.max_log_sigma_delta * torch.tanh(residuals[:, 0])
        delta_sigma_y = self.max_log_sigma_delta * torch.tanh(residuals[:, 1])
        delta_rho = self.max_rho_delta * torch.tanh(residuals[:, 2])
        delta_opacity = self.max_opacity_logit_delta * torch.tanh(residuals[:, 3])

        sigma_x = sigma_x.clamp_min(0.05) * torch.exp(delta_sigma_x)
        sigma_y = sigma_y.clamp_min(0.05) * torch.exp(delta_sigma_y)
        rho = (rho + delta_rho).clamp(-0.95, 0.95)

        opacity = opacity.clamp(1e-4, 1 - 1e-4)
        opacity = torch.sigmoid(torch.logit(opacity) + delta_opacity)

        patch_count = logits.shape[0]
        sigma_x = sigma_x.reshape(patch_count, -1)
        sigma_y = sigma_y.reshape(patch_count, -1)
        opacity = opacity.reshape(patch_count, -1)
        rho = rho.reshape(patch_count, -1)
        self.last_gaussian_parameters = tuple(
            value.detach() for value in (sigma_x, sigma_y, rho, opacity)
        )
        return sigma_x, sigma_y, opacity, rho

    def query_rgb(self, coord, scale, cell=None):
        feat = self.feat[:, :8, :, :]
        lr_feat = self.feat[:, 8:, :, :]
        logits = self.logits
        feat_size = feat.shape
        feat_device = feat.device

        scale_values = scale.reshape(-1).to(device=feat_device, dtype=feat.dtype)
        raster_scale = float(scale_values[0])
        hr_h = round(feat.shape[-2] * raster_scale)
        hr_w = round(feat.shape[-1] * raster_scale)

        num_kernels_row = math.ceil(feat_size[-2] / self.row)
        num_kernels_column = math.ceil(feat_size[-1] / self.column)
        upsampled_size = (
            num_kernels_row * self.row,
            num_kernels_column * self.column,
        )
        upsampled_feat = F.interpolate(
            feat, size=upsampled_size, mode='bicubic', align_corners=False
        )
        upsampled_logits = F.interpolate(
            logits, size=upsampled_size, mode='bicubic', align_corners=False
        )
        unfold = nn.Unfold(
            kernel_size=(self.row, self.column),
            stride=(self.row, self.column),
        )
        unfolded_feature = unfold(upsampled_feat)
        unfolded_logits = unfold(upsampled_logits)
        patch_per_image = unfolded_feature.shape[-1]
        patch_count = feat_size[0] * patch_per_image

        patch_features = unfolded_feature.transpose(1, 2).reshape(
            patch_count, feat_size[1], self.row, self.column
        )
        patch_logits = unfolded_logits.transpose(1, 2).reshape(
            patch_count, logits.shape[1], self.row, self.column
        )
        patch_scales = scale_values.unsqueeze(1).expand(
            -1, patch_per_image
        ).reshape(-1)

        coords = generate_meshgrid(self.row, self.column)
        point_count = self.row * self.column
        colors, normalized_coords = fetching_features_from_tensor(
            patch_features, coords
        )
        sigma_x, sigma_y, opacity, rho = self.local_gaussian_parameters(
            patch_logits, patch_features, patch_scales
        )

        covariance = torch.zeros(
            patch_count, point_count, 2, 2,
            dtype=feat.dtype,
            device=feat_device,
        )
        covariance[..., 0, 0] = sigma_x.pow(2) + 1e-5
        covariance[..., 1, 1] = sigma_y.pow(2) + 1e-5
        covariance[..., 0, 1] = rho * sigma_x * sigma_y
        covariance[..., 1, 0] = covariance[..., 0, 1]
        inverse_covariance = torch.inverse(covariance)

        axis = torch.linspace(-5, 5, self.kernel_size, device=feat_device)
        yy, xx = torch.meshgrid(axis, axis)
        xy = torch.stack((xx, yy), dim=-1)
        exponent = torch.einsum(
            'hwi,pnij,hwj->pnhw',
            xy,
            -0.5 * inverse_covariance,
            xy,
        )
        kernels = torch.exp(exponent)
        kernels = kernels / kernels.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)

        channel_count = patch_features.shape[1]
        kernel_images = kernels.unsqueeze(2).expand(
            -1, -1, channel_count, -1, -1
        ).reshape(
            patch_count * point_count,
            channel_count,
            self.kernel_size,
            self.kernel_size,
        ).contiguous()

        patch_hr_h = round(self.row * raster_scale)
        patch_hr_w = round(self.column * raster_scale)
        pad_h = patch_hr_h - self.kernel_size
        pad_w = patch_hr_w - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError(
                'Kernel size must not exceed the scaled local patch size.'
            )
        padding = (
            pad_w // 2,
            pad_w // 2 + pad_w % 2,
            pad_h // 2,
            pad_h // 2 + pad_h % 2,
        )
        kernel_images = F.pad(kernel_images, padding, mode='constant', value=0)

        theta = torch.zeros(
            patch_count, point_count, 2, 3,
            dtype=feat.dtype,
            device=feat_device,
        )
        theta[:, :, 0, 0] = 1
        theta[:, :, 1, 1] = 1
        theta[:, :, :, 2] = normalized_coords.unsqueeze(0)
        grid = F.affine_grid(
            theta.reshape(-1, 2, 3),
            size=[
                patch_count * point_count,
                channel_count,
                patch_hr_h,
                patch_hr_w,
            ],
            align_corners=True,
        )
        translated_kernels = F.grid_sample(
            kernel_images,
            grid,
            align_corners=True,
        ).reshape(
            patch_count,
            point_count,
            channel_count,
            patch_hr_h,
            patch_hr_w,
        )

        colors = colors * opacity.unsqueeze(-1)
        local_images = (
            colors.unsqueeze(-1).unsqueeze(-1) * translated_kernels
        ).sum(dim=1).clamp_(0, 1)

        fold = nn.Fold(
            output_size=(
                patch_hr_h * num_kernels_row,
                patch_hr_w * num_kernels_column,
            ),
            kernel_size=(patch_hr_h, patch_hr_w),
            stride=(patch_hr_h, patch_hr_w),
        )
        local_images = local_images.reshape(
            feat_size[0],
            patch_per_image,
            feat_size[1] * patch_hr_h * patch_hr_w,
        ).transpose(1, 2)
        final_image = fold(local_images)
        final_image = F.interpolate(
            final_image,
            size=(hr_h, hr_w),
            mode='bicubic',
            align_corners=False,
        )
        lr_feat = F.interpolate(
            lr_feat,
            size=(hr_h, hr_w),
            mode='bicubic',
            align_corners=False,
        )
        final_image = torch.cat((final_image, lr_feat), dim=1)

        coefficient = self.coef(final_image)
        frequency = self.freq(final_image)
        query_coord = coord.clone()
        query_coefficient = F.grid_sample(
            coefficient,
            query_coord.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        query_frequency = F.grid_sample(
            frequency,
            query_coord.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        query_feature_coord = F.grid_sample(
            self.feat_coord,
            query_coord.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

        relative_coord = coord - query_feature_coord
        relative_coord[:, :, 0] *= feat.shape[-2]
        relative_coord[:, :, 1] *= feat.shape[-1]
        relative_cell = cell.clone()
        relative_cell[:, :, 0] *= feat.shape[-2]
        relative_cell[:, :, 1] *= feat.shape[-1]

        batch_size, query_count = coord.shape[:2]
        query_frequency = torch.stack(
            torch.split(query_frequency, 2, dim=-1), dim=-1
        )
        query_frequency = torch.mul(
            query_frequency, relative_coord.unsqueeze(-1)
        ).sum(dim=-2)
        query_frequency += self.phase(
            relative_cell.reshape(batch_size * query_count, -1)
        ).reshape(batch_size, query_count, -1)
        query_frequency = torch.cat((
            torch.cos(math.pi * query_frequency),
            torch.sin(math.pi * query_frequency),
        ), dim=-1)

        decoder_input = torch.mul(query_coefficient, query_frequency)
        return self.dec(
            decoder_input.contiguous().reshape(batch_size * query_count, -1)
        ).reshape(batch_size, query_count, -1)
