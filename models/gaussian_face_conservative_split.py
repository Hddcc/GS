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


class ConservativeSplitPredictor(nn.Module):
    """Predict a moment-conserving pair of sub-Gaussians per LR point."""

    def __init__(
            self,
            feature_dim,
            hidden_dim=48,
            scale_min=1.5,
            scale_max=8.0,
            separation_init=0.5):
        super().__init__()
        if not 0 < separation_init < 1:
            raise ValueError('separation_init must be between 0 and 1.')
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError('Expected 0 < scale_min < scale_max.')

        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ]) / 8
        sobel_y = sobel_x.t().contiguous()
        self.register_buffer(
            'sobel_filters',
            torch.stack((sobel_x, sobel_y)).unsqueeze(1),
        )
        self.log_scale_min = math.log(scale_min)
        self.log_scale_range = math.log(scale_max) - self.log_scale_min
        self.feature_dim = feature_dim

        input_dim = feature_dim + 3 + 4
        output_dim = 2 + 1 + feature_dim + 1
        self.body = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        with torch.no_grad():
            self.body[-1].bias[2] = math.log(
                separation_init / (1 - separation_init)
            )
            # Nonzero paired feature residuals let earlier predictor layers
            # receive gradients from the first optimization step.
            nn.init.normal_(
                self.body[-1].weight[3:3 + feature_dim],
                mean=0.0,
                std=0.02,
            )

    def scale_features(self, patch_scales, dtype):
        normalized = (
            (patch_scales.to(dtype=dtype).clamp_min(1e-6).log()
             - self.log_scale_min)
            / self.log_scale_range
        ).clamp(0, 1)
        return torch.stack((
            normalized,
            normalized.square(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ), dim=-1)

    def structure_direction(self, patch_features):
        gray = patch_features.mean(dim=1, keepdim=True)
        gradients = F.conv2d(
            gray,
            self.sobel_filters.to(dtype=gray.dtype),
            padding=1,
        )
        energy = torch.linalg.vector_norm(
            gradients, dim=1, keepdim=True
        )
        direction = gradients / energy.clamp_min(1e-6)
        return direction, energy

    def forward(self, patch_features, patch_scales):
        direction, energy = self.structure_direction(patch_features)
        normalized_energy = torch.log1p(
            energy / energy.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        )
        scale_embedding = self.scale_features(
            patch_scales, patch_features.dtype
        ).unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, patch_features.shape[-2], patch_features.shape[-1]
        )
        raw = self.body(torch.cat((
            patch_features,
            direction,
            normalized_energy,
            scale_embedding,
        ), dim=1))

        direction_delta = 0.25 * torch.tanh(raw[:, :2])
        split_direction = direction + direction_delta
        split_direction = split_direction / torch.linalg.vector_norm(
            split_direction, dim=1, keepdim=True
        ).clamp_min(1e-6)
        separation = torch.sigmoid(raw[:, 2:3])
        feature_delta = raw[:, 3:3 + self.feature_dim]
        local_blend = torch.sigmoid(raw[:, -1:])
        return split_direction, separation, feature_delta, local_blend


@register('gaussian-splatter-face-conservative-split')
class FaceConservativeSplitGaussianSplatter(GaussianSplatter):
    """GaussianSR with paired, moment-conserving subpixel Gaussians."""

    def __init__(
            self,
            encoder_spec,
            dec_spec,
            kernel_size,
            hidden_dim=256,
            unfold_row=7,
            unfold_column=7,
            num_points=100,
            split_hidden_dim=48,
            scale_min=1.5,
            scale_max=8.0,
            max_offset=2.0,
            max_feature_delta=0.5,
            split_gate_init=0.3,
            separation_init=0.5):
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
        )
        if max_offset <= 0 or max_feature_delta <= 0:
            raise ValueError('Split bounds must be positive.')
        if not 0 < split_gate_init < 1:
            raise ValueError('split_gate_init must be between 0 and 1.')

        split_feature_dim = 8
        self.split_predictor = ConservativeSplitPredictor(
            feature_dim=split_feature_dim,
            hidden_dim=split_hidden_dim,
            scale_min=scale_min,
            scale_max=scale_max,
            separation_init=separation_init,
        )
        self.max_offset = max_offset
        self.max_feature_delta = max_feature_delta
        gate_logit = math.log(split_gate_init / (1 - split_gate_init))
        self.split_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        self.last_split_offset = None
        self.last_split_blend = None
        self.last_split_feature_delta = None
        self.last_center_conservation_error = None
        self.last_feature_conservation_error = None

    def local_base_parameters(self, patch_logits):
        probabilities = patch_logits.permute(0, 2, 3, 1)
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
        patch_count = patch_logits.shape[0]
        return (
            sigma_x.reshape(patch_count, -1).clamp_min(0.05),
            sigma_y.reshape(patch_count, -1).clamp_min(0.05),
            opacity.reshape(patch_count, -1),
            rho.reshape(patch_count, -1).clamp(-0.95, 0.95),
        )

    def split_parameters(self, patch_features, patch_scales):
        direction, separation, raw_delta, local_blend = self.split_predictor(
            patch_features, patch_scales
        )
        offset = self.max_offset * separation * direction
        feature_delta = self.max_feature_delta * torch.tanh(raw_delta)
        blend = torch.sigmoid(self.split_gate_logit) * local_blend

        patch_count = patch_features.shape[0]
        offset = offset.permute(0, 2, 3, 1).reshape(patch_count, -1, 2)
        feature_delta = feature_delta.permute(0, 2, 3, 1).reshape(
            patch_count, -1, patch_features.shape[1]
        )
        blend = blend.reshape(patch_count, -1)

        center_error = (0.5 * offset + 0.5 * -offset).abs().amax()
        feature_error = (
            0.5 * feature_delta + 0.5 * -feature_delta
        ).abs().amax()
        self.last_split_offset = offset.detach()
        self.last_split_blend = blend.detach()
        self.last_split_feature_delta = feature_delta.detach()
        self.last_center_conservation_error = center_error.detach()
        self.last_feature_conservation_error = feature_error.detach()
        return offset, feature_delta, blend

    @staticmethod
    def gaussian_kernels(xy, inverse_covariance, offset):
        displacement = xy.view(1, 1, *xy.shape) - offset.unsqueeze(2).unsqueeze(2)
        exponent = torch.einsum(
            'pnhwi,pnij,pnhwj->pnhw',
            displacement,
            -0.5 * inverse_covariance,
            displacement,
        )
        kernels = torch.exp(exponent)
        return kernels / kernels.amax(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)

    def query_rgb(self, coord, scale, cell=None):
        if cell is None:
            raise ValueError('cell is required for conservative splitting.')
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
        sigma_x, sigma_y, opacity, rho = self.local_base_parameters(
            patch_logits
        )
        offset, feature_delta, blend = self.split_parameters(
            patch_features, patch_scales
        )

        covariance = torch.zeros(
            patch_count, point_count, 2, 2,
            dtype=feat.dtype,
            device=feat_device,
        )
        covariance[..., 0, 0] = sigma_x.square() + 1e-5
        covariance[..., 1, 1] = sigma_y.square() + 1e-5
        covariance[..., 0, 1] = rho * sigma_x * sigma_y
        covariance[..., 1, 0] = covariance[..., 0, 1]
        inverse_covariance = torch.inverse(covariance)

        axis = torch.linspace(-5, 5, self.kernel_size, device=feat_device)
        yy, xx = torch.meshgrid(axis, axis)
        xy = torch.stack((xx, yy), dim=-1)
        zero_offset = torch.zeros_like(offset)
        kernels = torch.stack((
            self.gaussian_kernels(xy, inverse_covariance, zero_offset),
            self.gaussian_kernels(xy, inverse_covariance, offset),
            self.gaussian_kernels(xy, inverse_covariance, -offset),
        ), dim=2)

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
        kernel_images = F.pad(
            kernels.reshape(-1, 1, self.kernel_size, self.kernel_size),
            padding,
            mode='constant',
            value=0,
        )

        component_count = 3
        theta = torch.zeros(
            patch_count, point_count, component_count, 2, 3,
            dtype=feat.dtype,
            device=feat_device,
        )
        theta[..., 0, 0] = 1
        theta[..., 1, 1] = 1
        theta[..., :, 2] = normalized_coords.view(1, point_count, 1, 2)
        grid = F.affine_grid(
            theta.reshape(-1, 2, 3),
            size=[
                patch_count * point_count * component_count,
                1,
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
            component_count,
            patch_hr_h,
            patch_hr_w,
        )

        parent_weight = 1 - blend
        child_weight = 0.5 * blend
        component_features = torch.stack((
            parent_weight.unsqueeze(-1) * colors,
            child_weight.unsqueeze(-1) * (colors + feature_delta),
            child_weight.unsqueeze(-1) * (colors - feature_delta),
        ), dim=2)
        component_features = component_features * opacity.unsqueeze(-1).unsqueeze(-1)
        local_images = torch.einsum(
            'pnkc,pnkhw->pchw', component_features, translated_kernels
        ).clamp_(0, 1)

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
        query_grid = coord.flip(-1).unsqueeze(1)
        query_coefficient = F.grid_sample(
            coefficient,
            query_grid,
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        query_frequency = F.grid_sample(
            frequency,
            query_grid,
            mode='nearest',
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
        relative_coord[:, :, 0] *= feat.shape[-2]
        relative_coord[:, :, 1] *= feat.shape[-1]
        relative_cell = cell.clone()
        relative_cell[:, :, 0] *= feat.shape[-2]
        relative_cell[:, :, 1] *= feat.shape[-1]

        batch_size, query_count = coord.shape[:2]
        query_frequency = torch.stack(
            torch.split(query_frequency, 2, dim=-1), dim=-1
        )
        query_frequency = (
            query_frequency * relative_coord.unsqueeze(-1)
        ).sum(dim=-2)
        query_frequency += self.phase(
            relative_cell.reshape(batch_size * query_count, -1)
        ).reshape(batch_size, query_count, -1)
        query_frequency = torch.cat((
            torch.cos(math.pi * query_frequency),
            torch.sin(math.pi * query_frequency),
        ), dim=-1)
        decoder_input = query_coefficient * query_frequency
        return self.dec(
            decoder_input.contiguous().reshape(batch_size * query_count, -1)
        ).reshape(batch_size, query_count, -1)
