import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.gaussian import GaussianSplatter


def make_mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


@register('gaussian-splatter-face-scale-moe')
class FaceScaleMoEGaussianSplatter(GaussianSplatter):
    """GaussianSR with scale-conditioned query-wise RGB residual experts."""

    def __init__(
            self,
            encoder_spec,
            dec_spec,
            kernel_size,
            hidden_dim=256,
            unfold_row=7,
            unfold_column=7,
            num_points=100,
            expert_feature_dim=48,
            expert_hidden_dim=64,
            num_experts=3,
            scale_min=1.5,
            scale_max=8.0,
            router_temperature=0.7,
            scale_prior_strength=1.0,
            router_content_strength=None,
            balance_loss_weight=0.01,
            entropy_loss_weight=0.001,
            prior_loss_weight=0.0,
            experts_receive_scale=True,
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
        if num_experts < 2:
            raise ValueError('num_experts must be at least 2.')
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError('Expected 0 < scale_min < scale_max.')
        if router_temperature <= 0:
            raise ValueError('router_temperature must be positive.')
        if scale_prior_strength < 0:
            raise ValueError('scale_prior_strength must be non-negative.')
        if router_content_strength is not None and router_content_strength <= 0:
            raise ValueError('router_content_strength must be positive or null.')
        if (
                balance_loss_weight < 0
                or entropy_loss_weight < 0
                or prior_loss_weight < 0):
            raise ValueError('Auxiliary loss weights must be non-negative.')
        if max_rgb_residual <= 0:
            raise ValueError('max_rgb_residual must be positive.')
        if not 0 < residual_gate_init < 1:
            raise ValueError('residual_gate_init must be between 0 and 1.')

        self.context_encoder = nn.Sequential(
            nn.Conv2d(self.encoder.out_dim, expert_feature_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(expert_feature_dim, expert_feature_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        scale_feature_dim = 4
        local_query_dim = expert_feature_dim + 2 + 2 + 2
        router_query_dim = local_query_dim + scale_feature_dim
        expert_query_dim = (
            router_query_dim if experts_receive_scale else local_query_dim
        )
        self.router = make_mlp(
            router_query_dim, expert_hidden_dim, num_experts
        )
        self.experts = nn.ModuleList([
            make_mlp(expert_query_dim, expert_hidden_dim, 3)
            for _ in range(num_experts)
        ])

        # Start from the continuous scale prior and near-zero RGB corrections.
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        for expert in self.experts:
            nn.init.normal_(expert[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(expert[-1].bias)

        self.num_experts = num_experts
        self.log_scale_min = math.log(scale_min)
        self.log_scale_range = math.log(scale_max) - self.log_scale_min
        self.router_temperature = router_temperature
        self.scale_prior_strength = scale_prior_strength
        self.router_content_strength = router_content_strength
        self.balance_loss_weight = balance_loss_weight
        self.entropy_loss_weight = entropy_loss_weight
        self.prior_loss_weight = prior_loss_weight
        self.experts_receive_scale = experts_receive_scale
        self.max_rgb_residual = max_rgb_residual
        self.register_buffer(
            'expert_scale_centers', torch.linspace(0, 1, num_experts)
        )
        gate_logit = math.log(residual_gate_init / (1 - residual_gate_init))
        self.residual_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        self.expert_context = None
        self.last_router_weights = None
        self.last_rgb_residual = None
        self._training_diagnostics = None

    def gen_feat(self, inp):
        feature, logits = super().gen_feat(inp)
        self.expert_context = self.context_encoder(feature)
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
        encoded = torch.stack((
            normalized,
            normalized.square(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ), dim=-1)
        return (
            encoded.unsqueeze(1).expand(-1, query_count, -1),
            normalized.view(batch_size, 1, 1).expand(-1, query_count, -1),
        )

    def query_rgb(self, coord, scale, cell=None):
        base_prediction = super().query_rgb(coord, scale, cell)
        if cell is None:
            raise ValueError('cell is required for scale-conditioned experts.')

        query_grid = coord.flip(-1).unsqueeze(1)
        query_context = F.grid_sample(
            self.expert_context,
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
        scale_feature, normalized_scale = self.scale_features(
            scale,
            batch_size,
            query_count,
            query_context.dtype,
            query_context.device,
        )
        local_query_input = torch.cat((
            query_context,
            coord,
            relative_coord,
            relative_cell,
        ), dim=-1)
        router_input = torch.cat((
            local_query_input,
            scale_feature,
        ), dim=-1)
        expert_input = (
            router_input if self.experts_receive_scale else local_query_input
        )

        local_router_logits = self.router(router_input)
        if self.router_content_strength is not None:
            local_router_logits = (
                self.router_content_strength * torch.tanh(local_router_logits)
            )
        centers = self.expert_scale_centers.to(dtype=query_context.dtype)
        scale_prior_logits = -self.scale_prior_strength * (
            normalized_scale - centers.view(1, 1, -1)
        ).square() / self.router_temperature
        prior_weights = torch.softmax(scale_prior_logits, dim=-1)
        router_weights = torch.softmax(
            local_router_logits + scale_prior_logits, dim=-1
        )

        expert_predictions = torch.stack([
            expert(expert_input) for expert in self.experts
        ], dim=-2)
        residual = (
            expert_predictions * router_weights.unsqueeze(-1)
        ).sum(dim=-2)
        residual = (
            self.max_rgb_residual
            * torch.sigmoid(self.residual_gate_logit)
            * torch.tanh(residual)
        )

        usage = router_weights.mean(dim=(0, 1))
        entropy = -(
            router_weights
            * router_weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        prior_kl = (
            router_weights
            * (
                router_weights.clamp_min(1e-8).log()
                - prior_weights.clamp_min(1e-8).log()
            )
        ).sum(dim=-1).mean()
        self._training_diagnostics = {
            'router_usage': usage.unsqueeze(0),
            'router_prior_usage': prior_weights.mean(dim=(0, 1)).unsqueeze(0),
            'router_entropy': entropy.unsqueeze(0),
            'router_prior_kl': prior_kl.unsqueeze(0),
            'residual_abs_mean': residual.abs().mean().unsqueeze(0),
        }
        self.last_router_weights = router_weights.detach()
        self.last_rgb_residual = residual.detach()
        return base_prediction + residual

    def forward(self, inp, coord, scale, cell=None):
        self.gen_feat(inp)
        prediction = self.query_rgb(coord, scale, cell)
        if self.training:
            diagnostics = self._training_diagnostics
            self._training_diagnostics = None
            return prediction, diagnostics
        return prediction


register('gaussian-splatter-face-scale-moe-v2')(
    FaceScaleMoEGaussianSplatter
)
