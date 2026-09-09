import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models import register
from models.gaussian import fetching_features_from_tensor, generate_meshgrid
from models.gaussian_memory_efficient import MemoryEfficientGaussianSplatter


@register('gaussian-splatter-memory-efficient-m4')
class MultiPrimitiveGaussianSplatter(MemoryEfficientGaussianSplatter):
    """Four independently trainable subpixel primitives per LR position."""

    def __init__(self, *args, primitive_count=4, residual_limit=0.1,
                 offset_residual_limit=0.25, covariance_log_limit=0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if primitive_count != 4:
            raise ValueError('The preregistered model requires 4 primitives.')
        if self.gaussian_channels != 8:
            raise ValueError('The preregistered model requires 8 Gaussian channels.')
        self.primitive_count = primitive_count
        self.residual_limit = float(residual_limit)
        self.offset_residual_limit = float(offset_residual_limit)
        self.covariance_log_limit = float(covariance_log_limit)
        self.primitive_residual_head = nn.Conv2d(
            self.encoder.out_dim,
            primitive_count * self.gaussian_channels,
            kernel_size=1,
        )
        nn.init.normal_(self.primitive_residual_head.weight, std=1e-3)
        nn.init.zeros_(self.primitive_residual_head.bias)
        self.offset_delta = nn.Parameter(torch.zeros(primitive_count, 2))
        self.covariance_scale_delta = nn.Parameter(
            torch.zeros(primitive_count, 2)
        )
        initial_opacity = math.log(0.25 / 0.75)
        self.opacity_logit = nn.Parameter(
            torch.full((primitive_count,), initial_opacity)
        )
        self.register_buffer(
            'base_offsets',
            torch.tensor([
                [-0.25, -0.25],
                [-0.25, 0.25],
                [0.25, -0.25],
                [0.25, 0.25],
            ]),
        )
        self.primitive_residual = None
        self.last_primitive_diagnostics = None

    def gen_feat(self, inp):
        feature, logits = super().gen_feat(inp)
        self.primitive_residual = self.primitive_residual_head(feature)
        return feature, logits

    def primitive_offsets(self):
        return self.base_offsets + self.offset_residual_limit * torch.tanh(
            self.offset_delta
        )

    def covariance_scales(self):
        return torch.exp(
            self.covariance_log_limit
            * torch.tanh(self.covariance_scale_delta)
        )

    def opacity_weights(self):
        return torch.sigmoid(self.opacity_logit)

    def _render_patch_batch(self, feature, logits, primitive_residual, scale):
        device = feature.device
        patch_batch, channels, height, width = feature.shape
        point_coordinates = generate_meshgrid(height, width).to(device)
        point_count = height * width
        base_colors, _ = fetching_features_from_tensor(
            feature, point_coordinates
        )
        residual = primitive_residual.view(
            patch_batch,
            self.primitive_count,
            channels,
            height,
            width,
        ).permute(0, 3, 4, 1, 2).reshape(
            patch_batch, point_count, self.primitive_count, channels
        )
        residual = self.residual_limit * torch.tanh(residual)
        colors = base_colors.unsqueeze(2) + residual

        offsets = self.primitive_offsets()
        primitive_coordinates = (
            point_coordinates[:, None, :] + offsets[None, :, :]
        )
        coordinate_scale = feature.new_tensor([height, width])
        normalized_coordinates = (
            0.5 - primitive_coordinates / coordinate_scale
        ) * 2.0
        normalized_coordinates = normalized_coordinates.reshape(-1, 2)

        sigma_x, sigma_y, opacity, rho = self.weighted_gaussian_parameters(
            logits
        )
        covariance_scales = self.covariance_scales()
        sigma_x = (
            sigma_x[:, None] * covariance_scales[None, :, 0]
        ).reshape(-1, 1, 1)
        sigma_y = (
            sigma_y[:, None] * covariance_scales[None, :, 1]
        ).reshape(-1, 1, 1)
        rho = rho[:, None].expand(-1, self.primitive_count).reshape(-1, 1, 1)
        covariance = torch.stack(
            [
                torch.stack(
                    [sigma_x.square() + 1e-5, rho * sigma_x * sigma_y],
                    dim=-1,
                ),
                torch.stack(
                    [rho * sigma_x * sigma_y, sigma_y.square() + 1e-5],
                    dim=-1,
                ),
            ],
            dim=-2,
        )
        inverse_covariance = torch.inverse(covariance)

        axis = torch.linspace(-5.0, 5.0, steps=self.kernel_size, device=device)
        xx = axis.view(-1, 1).expand(-1, self.kernel_size)
        yy = axis.view(1, -1).expand(self.kernel_size, -1)
        xy = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        exponent = torch.einsum(
            'b...i,b...ij,b...j->b...',
            xy,
            -0.5 * inverse_covariance,
            xy,
        )
        kernel = torch.exp(exponent) / (
            2 * torch.tensor(np.pi, device=device)
            * torch.sqrt(torch.det(covariance)).view(-1, 1, 1)
        )
        kernel = kernel / kernel.amax(dim=(-2, -1), keepdim=True)

        primitive_count = point_count * self.primitive_count
        kernel = kernel.repeat(1, channels, 1).contiguous().view(
            primitive_count, channels, self.kernel_size, self.kernel_size
        )
        patch_h = round(height * scale)
        patch_w = round(width * scale)
        pad_h = patch_h - self.kernel_size
        pad_w = patch_w - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError('Kernel size must not exceed the HR patch size.')
        kernel = F.pad(
            kernel,
            (
                pad_w // 2,
                pad_w // 2 + pad_w % 2,
                pad_h // 2,
                pad_h // 2 + pad_h % 2,
            ),
            'constant',
            0,
        )
        opacity_weights = self.opacity_weights()
        primitive_opacity = (
            opacity[:, None] * opacity_weights[None, :]
        ).reshape(1, primitive_count, 1)
        colors = colors.reshape(
            patch_batch, primitive_count, channels
        ) * primitive_opacity

        output = feature.new_zeros(patch_batch, channels, patch_h, patch_w)
        self.last_baseline_layer_elements = (
            patch_batch * primitive_count * channels * patch_h * patch_w
        )
        self.last_peak_layer_elements = 0
        for first in range(0, primitive_count, self.primitive_chunk_size):
            stop = min(first + self.primitive_chunk_size, primitive_count)
            chunk_count = stop - first
            theta = torch.zeros(
                patch_batch,
                chunk_count,
                2,
                3,
                dtype=torch.float32,
                device=device,
            )
            theta[:, :, 0, 0] = 1.0
            theta[:, :, 1, 1] = 1.0
            theta[:, :, :, 2] = normalized_coordinates[first:stop]
            grid = F.affine_grid(
                theta.view(-1, 2, 3),
                size=[
                    patch_batch * chunk_count,
                    channels,
                    patch_h,
                    patch_w,
                ],
                align_corners=True,
            ).contiguous()
            chunk_kernel = kernel[first:stop]
            chunk_colors = colors[:, first:stop]

            def render_chunk(kernel_input, color_input, grid_input):
                translated = F.grid_sample(
                    kernel_input.repeat(
                        patch_batch, 1, 1, 1
                    ).contiguous(),
                    grid_input,
                    align_corners=True,
                ).view(
                    patch_batch, chunk_count, channels, patch_h, patch_w
                )
                return (
                    color_input.unsqueeze(-1).unsqueeze(-1) * translated
                ).sum(dim=1)

            if self.training and torch.is_grad_enabled():
                contribution = checkpoint(
                    render_chunk,
                    chunk_kernel,
                    chunk_colors,
                    grid,
                    use_reentrant=False,
                )
            else:
                contribution = render_chunk(
                    chunk_kernel, chunk_colors, grid
                )
            output = output + contribution
            self.last_peak_layer_elements = max(
                self.last_peak_layer_elements,
                patch_batch * chunk_count * channels * patch_h * patch_w,
            )

        branch_response = residual.abs().mean(dim=(0, 1, 3))
        self.last_primitive_diagnostics = {
            'branch_response': branch_response.detach(),
            'residual_mean_abs': residual.detach().abs().mean(),
            'offsets': offsets.detach(),
            'covariance_scales': covariance_scales.detach(),
            'opacity_weights': opacity_weights.detach(),
            'unclamped_output_mean_abs': output.detach().abs().mean(),
        }
        return output.clamp(0, 1)

    def query_rgb(self, coord, scale, cell=None):
        if cell is None:
            raise ValueError('cell is required.')
        scale_value = float(scale[0])
        if not torch.allclose(
                scale, scale.new_full(scale.shape, scale_value),
                rtol=0, atol=1e-6):
            raise ValueError('All images in a batch must share one scale.')
        if self.primitive_residual is None:
            raise RuntimeError('Call gen_feat before query_rgb.')

        feature = self.feat[:, :self.gaussian_channels]
        bypass = self.feat[:, self.gaussian_channels:]
        feature_size = feature.shape
        hr_h = round(feature.shape[-2] * scale_value)
        hr_w = round(feature.shape[-1] * scale_value)
        patch_rows = math.ceil(feature_size[-2] / self.row)
        patch_columns = math.ceil(feature_size[-1] / self.column)
        padded_size = (patch_rows * self.row, patch_columns * self.column)
        padded_feature = F.interpolate(
            feature, size=padded_size, mode='bicubic', align_corners=False
        )
        padded_logits = F.interpolate(
            self.logits, size=padded_size, mode='bicubic', align_corners=False
        )
        padded_residual = F.interpolate(
            self.primitive_residual,
            size=padded_size,
            mode='bicubic',
            align_corners=False,
        )
        unfold = nn.Unfold(
            kernel_size=(self.row, self.column),
            stride=(self.row, self.column),
        )
        feature_columns = unfold(padded_feature)
        logits_columns = unfold(padded_logits)
        residual_columns = unfold(padded_residual)
        patch_count = feature_columns.shape[-1]
        patch_feature = feature_columns.transpose(1, 2).reshape(
            feature_size[0] * patch_count,
            feature_size[1],
            self.row,
            self.column,
        )
        patch_logits = logits_columns.transpose(1, 2).reshape(
            self.logits.shape[0] * patch_count,
            self.logits.shape[1],
            self.row,
            self.column,
        )
        patch_residual = residual_columns.transpose(1, 2).reshape(
            self.primitive_residual.shape[0] * patch_count,
            self.primitive_residual.shape[1],
            self.row,
            self.column,
        )
        patch_image = self._render_patch_batch(
            patch_feature, patch_logits, patch_residual, scale_value
        )

        patch_h = round(self.row * scale_value)
        patch_w = round(self.column * scale_value)
        fold = nn.Fold(
            output_size=(patch_h * patch_rows, patch_w * patch_columns),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )
        patch_image = patch_image.reshape(
            feature_size[0],
            patch_count,
            feature_size[1] * patch_h * patch_w,
        ).transpose(1, 2)
        rendered = fold(patch_image)
        rendered = F.interpolate(
            rendered, size=(hr_h, hr_w), mode='bicubic', align_corners=False
        )
        bypass = F.interpolate(
            bypass, size=(hr_h, hr_w), mode='bicubic', align_corners=False
        )
        full_feature = torch.cat((rendered, bypass), dim=1)

        coef = self.coef(full_feature)
        freq = self.freq(full_feature)
        sample_grid = coord.flip(-1).unsqueeze(1)
        q_coef = F.grid_sample(
            coef, sample_grid, mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        q_freq = F.grid_sample(
            freq, sample_grid, mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(
            self.feat_coord,
            sample_grid,
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feature.shape[-2]
        rel_coord[:, :, 1] *= feature.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feature.shape[-2]
        rel_cell[:, :, 1] *= feature.shape[-1]
        batch_size, query_count = coord.shape[:2]
        q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
        q_freq = torch.sum(q_freq * rel_coord.unsqueeze(-1), dim=-2)
        q_freq += self.phase(
            rel_cell.view(batch_size * query_count, -1)
        ).view(batch_size, query_count, -1)
        q_freq = torch.cat(
            (torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1
        )
        decoder_input = q_coef * q_freq
        return self.dec(decoder_input.contiguous().view(
            batch_size * query_count, -1
        )).view(batch_size, query_count, -1)
