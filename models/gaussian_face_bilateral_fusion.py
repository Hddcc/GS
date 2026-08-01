import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import GaussianSplatter


@register('gaussian-splatter-face-bilateral-fusion')
class FaceBilateralFusionGaussianSplatter(GaussianSplatter):
    """GaussianSR with confidence-gated bilateral feature fusion."""

    def __init__(
            self,
            encoder_spec,
            dec_spec,
            kernel_size,
            hidden_dim=256,
            unfold_row=7,
            unfold_column=7,
            num_points=100,
            fusion_hidden_dim=64,
            confidence_hidden_dim=32,
            asymmetry_temperature=0.5,
            fusion_gate_init=0.25):
        super().__init__(
            encoder_spec=encoder_spec,
            dec_spec=dec_spec,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            unfold_row=unfold_row,
            unfold_column=unfold_column,
            num_points=num_points,
        )
        if asymmetry_temperature <= 0:
            raise ValueError('asymmetry_temperature must be positive.')
        if not 0 < fusion_gate_init < 1:
            raise ValueError('fusion_gate_init must be between 0 and 1.')

        feature_dim = self.encoder.out_dim
        paired_dim = feature_dim * 3
        self.confidence_head = nn.Sequential(
            nn.Conv2d(paired_dim + 1, confidence_hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(confidence_hidden_dim, 1, 3, padding=1),
        )
        self.fusion_head = nn.Sequential(
            nn.Conv2d(paired_dim, fusion_hidden_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_hidden_dim, feature_dim, 3, padding=1),
        )

        # Begin close to the baseline while retaining gradients for both heads.
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)
        nn.init.normal_(self.fusion_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fusion_head[-1].bias)

        self.asymmetry_temperature = asymmetry_temperature
        gate_logit = math.log(fusion_gate_init / (1 - fusion_gate_init))
        self.fusion_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        self.last_bilateral_confidence = None
        self.last_bilateral_prior = None
        self.last_bilateral_residual = None

    def gen_feat(self, inp):
        feature, logits = super().gen_feat(inp)
        mirror_feature = torch.flip(feature, dims=(-1,))
        feature_difference = (feature - mirror_feature).abs()
        paired_feature = torch.cat((
            feature,
            mirror_feature,
            feature_difference,
        ), dim=1)

        mirror_input = torch.flip(inp, dims=(-1,))
        image_asymmetry = (inp - mirror_input).abs().mean(dim=1, keepdim=True)
        image_asymmetry = F.interpolate(
            image_asymmetry,
            size=feature.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        bilateral_prior = torch.exp(
            -image_asymmetry / self.asymmetry_temperature
        )

        confidence_logits = self.confidence_head(torch.cat((
            paired_feature,
            image_asymmetry,
        ), dim=1))
        # Both members of a mirrored pair must share the same confidence.
        confidence_logits = 0.5 * (
            confidence_logits + torch.flip(confidence_logits, dims=(-1,))
        )
        confidence = torch.sigmoid(confidence_logits) * bilateral_prior

        bilateral_delta = torch.tanh(self.fusion_head(paired_feature))
        bilateral_residual = (
            torch.sigmoid(self.fusion_gate_logit)
            * confidence
            * bilateral_delta
        )
        self.feat = feature + bilateral_residual

        self.last_bilateral_confidence = confidence.detach()
        self.last_bilateral_prior = bilateral_prior.detach()
        self.last_bilateral_residual = bilateral_residual.detach()
        return self.feat, logits
