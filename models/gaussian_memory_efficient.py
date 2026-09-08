import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from models import register
from models.gaussian import (
    GaussianSplatter,
    fetching_features_from_tensor,
    generate_meshgrid,
)


@register('gaussian-splatter-memory-efficient')
class MemoryEfficientGaussianSplatter(GaussianSplatter):
    """Numerically equivalent GaussianSR rasterization in primitive chunks."""

    def __init__(self, *args, primitive_chunk_size=7, gaussian_channels=8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if primitive_chunk_size < 1:
            raise ValueError('primitive_chunk_size must be positive.')
        if not 0 < gaussian_channels <= self.encoder.out_dim:
            raise ValueError('gaussian_channels must be within encoder channels.')
        self.primitive_chunk_size = int(primitive_chunk_size)
        self.gaussian_channels = int(gaussian_channels)
        self.last_peak_layer_elements = 0
        self.last_baseline_layer_elements = 0

    def _render_patch_batch(self, feature, logits, scale):
        """Render unfolded patches without materializing all primitive layers."""
        device = feature.device
        point_coordinates = generate_meshgrid(feature.shape[-2], feature.shape[-1])
        point_count = feature.shape[-2] * feature.shape[-1]
        colors, normalized_coordinates = fetching_features_from_tensor(
            feature, point_coordinates
        )
        sigma_x, sigma_y, opacity, rho = self.weighted_gaussian_parameters(
            logits
        )
        sigma_x = sigma_x.view(point_count, 1, 1)
        sigma_y = sigma_y.view(point_count, 1, 1)
        rho = rho.view(point_count, 1, 1)
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
        inverse_covariance = torch.inverse(covariance).to(device)

        start = torch.tensor([-5.0], device=device).view(-1, 1)
        end = torch.tensor([5.0], device=device).view(-1, 1)
        axis = start + (end - start) * torch.linspace(
            0, 1, steps=self.kernel_size, device=device
        )
        xx = axis.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        yy = axis.unsqueeze(1).expand(-1, self.kernel_size, -1)
        xy = torch.stack([xx, yy], dim=-1)
        exponent = torch.einsum(
            'b...i,b...ij,b...j->b...',
            xy,
            -0.5 * inverse_covariance,
            xy,
        )
        kernel = torch.exp(exponent) / (
            2 * torch.tensor(np.pi, device=device)
            * torch.sqrt(torch.det(covariance)).to(device).view(
                point_count, 1, 1
            )
        )
        kernel_max = kernel.max(dim=-1, keepdim=True)[0].max(
            dim=-2, keepdim=True
        )[0]
        kernel = kernel / kernel_max

        patch_batch, channels = feature.shape[:2]
        kernel = kernel.repeat(1, channels, 1).contiguous().view(
            point_count, channels, self.kernel_size, self.kernel_size
        )
        patch_h = round(feature.shape[-2] * scale)
        patch_w = round(feature.shape[-1] * scale)
        pad_h = patch_h - self.kernel_size
        pad_w = patch_w - self.kernel_size
        if pad_h < 0 or pad_w < 0:
            raise ValueError('Kernel size must not exceed the HR patch size.')
        padding = (
            pad_w // 2,
            pad_w // 2 + pad_w % 2,
            pad_h // 2,
            pad_h // 2 + pad_h % 2,
        )
        kernel = F.pad(kernel, padding, 'constant', 0)
        colors = colors * opacity.to(device).unsqueeze(-1).expand(
            patch_batch, -1, -1
        )

        output = feature.new_zeros(patch_batch, channels, patch_h, patch_w)
        self.last_baseline_layer_elements = (
            patch_batch * point_count * channels * patch_h * patch_w
        )
        self.last_peak_layer_elements = 0
        for first in range(0, point_count, self.primitive_chunk_size):
            stop = min(first + self.primitive_chunk_size, point_count)
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
                    render_chunk, chunk_kernel, chunk_colors, grid,
                    use_reentrant=False,
                )
            else:
                contribution = render_chunk(chunk_kernel, chunk_colors, grid)
            output = output + contribution
            self.last_peak_layer_elements = max(
                self.last_peak_layer_elements,
                patch_batch * chunk_count * channels * patch_h * patch_w,
            )
        return output.clamp(0, 1)

    def query_rgb(self, coord, scale, cell=None):
        if cell is None:
            raise ValueError('cell is required.')
        scale_value = float(scale[0])
        if not torch.allclose(
                scale, scale.new_full(scale.shape, scale_value),
                rtol=0, atol=1e-6):
            raise ValueError('All images in a batch must share one scale.')

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
        unfold = nn.Unfold(
            kernel_size=(self.row, self.column),
            stride=(self.row, self.column),
        )
        feature_columns = unfold(padded_feature)
        logits_columns = unfold(padded_logits)
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
        patch_image = self._render_patch_batch(
            patch_feature, patch_logits, scale_value
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
        if bypass.shape[1] == 0:
            full_feature = rendered
        else:
            bypass = F.interpolate(
                bypass, size=(hr_h, hr_w), mode='bicubic',
                align_corners=False,
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
            self.feat_coord, sample_grid, mode='nearest', align_corners=False
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
        q_freq += self.phase(rel_cell.view(batch_size * query_count, -1)).view(
            batch_size, query_count, -1
        )
        q_freq = torch.cat(
            (torch.cos(np.pi * q_freq), torch.sin(np.pi * q_freq)), dim=-1
        )
        decoder_input = q_coef * q_freq
        return self.dec(decoder_input.contiguous().view(
            batch_size * query_count, -1
        )).view(batch_size, query_count, -1)
