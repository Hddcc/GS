import numpy as np
import torch
import torch.nn.functional as F

from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-metric-ensemble')
class FaceGaussianMetricEnsemble(FaceFrequencyResidualGaussianSplatter):
    """Blend four query decodes using learned Gaussian covariance distance."""

    def gaussian_parameter_maps(self):
        probabilities = self.logits
        sigma_x = (
            probabilities * self.sigma_x.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True).abs().clamp_min(1e-3)
        sigma_y = (
            probabilities * self.sigma_y.view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True).abs().clamp_min(1e-3)
        rho = (
            probabilities * self.rho[:, 0].view(1, -1, 1, 1)
        ).sum(dim=1, keepdim=True).clamp(-0.99, 0.99)
        return sigma_x, sigma_y, rho

    @staticmethod
    def sample_map(feature, grid):
        return F.grid_sample(
            feature,
            grid.flip(-1).unsqueeze(1),
            mode='nearest',
            align_corners=False,
        )[:, :, 0, :].permute(0, 2, 1)

    def decode_queries(self, coef, freq, coord, cell):
        sigma_x_map, sigma_y_map, rho_map = self.gaussian_parameter_maps()
        height, width = coef.shape[-2:]
        batch_size, query_count = coord.shape[:2]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= height
        rel_cell[:, :, 1] *= width
        phase = self.phase(rel_cell.view(batch_size * query_count, -1)).view(
            batch_size, query_count, -1
        )

        predictions = []
        metric_logits = []
        for offset_y in (-1, 1):
            for offset_x in (-1, 1):
                sample_coord = coord.clone()
                sample_coord[:, :, 0] += offset_y / height
                sample_coord[:, :, 1] += offset_x / width
                sample_coord.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_coef = self.sample_map(coef, sample_coord)
                q_freq = self.sample_map(freq, sample_coord)
                q_coord = self.sample_map(self.feat_coord, sample_coord)
                relative_coord = coord - q_coord
                relative_coord = relative_coord.clone()
                relative_coord[:, :, 0] *= height
                relative_coord[:, :, 1] *= width

                q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
                q_freq = torch.mul(q_freq, relative_coord.unsqueeze(-1))
                q_freq = torch.sum(q_freq, dim=-2) + phase
                q_freq = torch.cat((
                    torch.cos(np.pi * q_freq),
                    torch.sin(np.pi * q_freq),
                ), dim=-1)
                decoder_input = torch.mul(q_coef, q_freq)
                predictions.append(self.dec(
                    decoder_input.contiguous().view(
                        batch_size * query_count, -1
                    )
                ).view(batch_size, query_count, -1))

                sigma_x = self.sample_map(sigma_x_map, sample_coord)
                sigma_y = self.sample_map(sigma_y_map, sample_coord)
                rho = self.sample_map(rho_map, sample_coord)
                one_minus_rho_sq = (1 - rho.square()).clamp_min(1e-4)
                dy = relative_coord[:, :, 0:1]
                dx = relative_coord[:, :, 1:2]
                mahalanobis = (
                    dy.square() / sigma_y.square()
                    - 2 * rho * dx * dy / (sigma_x * sigma_y)
                    + dx.square() / sigma_x.square()
                ) / one_minus_rho_sq
                metric_logits.append(-0.5 * mahalanobis)

        ensemble_weights = F.softmax(
            torch.cat(metric_logits, dim=-1), dim=-1
        )
        prediction_stack = torch.stack(predictions, dim=-2)
        prediction = (
            prediction_stack * ensemble_weights.unsqueeze(-1)
        ).sum(dim=-2)
        self.last_ensemble_weights = ensemble_weights.detach()
        return prediction
