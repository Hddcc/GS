import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.edsr_geometry_appearance import GeometryAppearanceEncoder


class ScaleDepthRouterEncoder(GeometryAppearanceEncoder):
    """Route shallow, middle, and deep features by content and target scale."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            appearance_gate_init=0.25,
            routing_dim=32,
            scale_hidden_dim=16,
            scale_min=1.5,
            scale_max=8.0):
        super().__init__(
            args=args,
            n_class=n_class,
            shallow_depth=shallow_depth,
            middle_depth=middle_depth,
            geometry_dim=geometry_dim,
            fusion_dim=args.n_feats,
            appearance_gate_init=appearance_gate_init,
        )
        if routing_dim <= 0 or scale_hidden_dim <= 0:
            raise ValueError('Routing dimensions must be positive.')
        if scale_min <= 0 or scale_max <= scale_min:
            raise ValueError('Expected 0 < scale_min < scale_max.')

        # Replace the static concat-convolution fusion inherited above.
        self.appearance_fusion = nn.Identity()
        self.content_router = nn.Sequential(
            nn.Conv2d(3 * args.n_feats, routing_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(routing_dim, 3, 3, padding=1),
        )
        self.scale_router = nn.Sequential(
            nn.Linear(4, scale_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(scale_hidden_dim, 3),
        )
        nn.init.normal_(self.content_router[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.content_router[-1].bias)
        nn.init.normal_(self.scale_router[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.scale_router[-1].bias)

        self.log_scale_min = math.log(scale_min)
        self.log_scale_range = math.log(scale_max) - self.log_scale_min
        self.last_depth_weights = None
        self.last_depth_entropy = None
        self.last_scale_logits = None

    def scale_features(self, scale, batch_size, dtype, device):
        scale = scale.reshape(-1).to(device=device, dtype=dtype)
        if scale.numel() == 1:
            scale = scale.expand(batch_size)
        if scale.numel() != batch_size:
            raise ValueError('Expected one scale per batch sample.')
        normalized = (
            (scale.clamp_min(1e-6).log() - self.log_scale_min)
            / self.log_scale_range
        ).clamp(0, 1)
        return torch.stack((
            normalized,
            normalized.square(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ), dim=-1)

    def forward(self, inp, scale):
        shallow, middle, deep = self.hierarchical_features(inp)

        geometry_feature = torch.cat((
            self.geometry_shallow(shallow),
            self.geometry_middle(middle),
        ), dim=1)
        geometry_logits = self.geometry_head(geometry_feature)
        probabilities = self.geometry_probabilities(geometry_logits)

        routing_evidence = torch.cat((
            shallow - deep,
            middle - deep,
            shallow - middle,
        ), dim=1)
        content_logits = self.content_router(routing_evidence)
        scale_feature = self.scale_features(
            scale,
            inp.shape[0],
            content_logits.dtype,
            content_logits.device,
        )
        scale_logits = self.scale_router(scale_feature).unsqueeze(-1).unsqueeze(-1)
        depth_weights = F.softmax(content_logits + scale_logits, dim=1)
        centered_weights = depth_weights - (1.0 / 3.0)
        routed_delta = (
            centered_weights[:, 0:1] * shallow
            + centered_weights[:, 1:2] * middle
            + centered_weights[:, 2:3] * deep
        )
        appearance_residual = (
            torch.sigmoid(self.appearance_gate_logit)
            * routed_delta
        )
        appearance = deep + appearance_residual

        geometry_entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        depth_entropy = -(
            depth_weights.clamp_min(1e-8)
            * depth_weights.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        self.last_geometry_probabilities = probabilities.detach()
        self.last_geometry_entropy = geometry_entropy.detach()
        self.last_appearance_residual = appearance_residual.detach()
        self.last_depth_weights = depth_weights.detach()
        self.last_depth_entropy = depth_entropy.detach()
        self.last_scale_logits = scale_logits.detach()
        return appearance, probabilities


@register('edsr-face-scale-depth-router')
def make_scale_depth_router_encoder(
        n_resblocks=16,
        n_feats=64,
        res_scale=1,
        scale=2,
        no_upsampling=True,
        rgb_range=1,
        n_class=100,
        use_pretrained=False,
        shallow_depth=4,
        middle_depth=10,
        geometry_dim=32,
        appearance_gate_init=0.25,
        routing_dim=32,
        scale_hidden_dim=16,
        scale_min=1.5,
        scale_max=8.0):
    if use_pretrained:
        raise ValueError('Scale-depth router requires random initialization.')
    args = Namespace(
        n_resblocks=n_resblocks,
        n_feats=n_feats,
        res_scale=res_scale,
        scale=[scale],
        no_upsampling=no_upsampling,
        rgb_range=rgb_range,
        n_colors=3,
        pretrained_path=None,
    )
    return ScaleDepthRouterEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        appearance_gate_init=appearance_gate_init,
        routing_dim=routing_dim,
        scale_hidden_dim=scale_hidden_dim,
        scale_min=scale_min,
        scale_max=scale_max,
    )
