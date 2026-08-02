import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.edsr_geometry_appearance import GeometryAppearanceEncoder


class GeometryGuidedAppearanceEncoder(GeometryAppearanceEncoder):
    """Inject stable geometry evidence into appearance restoration."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            fusion_dim=64,
            appearance_gate_init=0.25,
            guidance_dim=32,
            guidance_gate_init=0.25):
        super().__init__(
            args=args,
            n_class=n_class,
            shallow_depth=shallow_depth,
            middle_depth=middle_depth,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            appearance_gate_init=appearance_gate_init,
        )
        if guidance_dim <= 0:
            raise ValueError('guidance_dim must be positive.')
        if not 0 < guidance_gate_init < 1:
            raise ValueError('guidance_gate_init must be between 0 and 1.')

        self.geometry_guidance = nn.Sequential(
            nn.Conv2d(n_class, guidance_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(guidance_dim, guidance_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.appearance_fusion = nn.Sequential(
            nn.Conv2d(3 * args.n_feats + guidance_dim, fusion_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_dim, args.n_feats, 3, padding=1),
        )
        nn.init.normal_(
            self.appearance_fusion[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.appearance_fusion[-1].bias)
        guidance_gate_logit = math.log(
            guidance_gate_init / (1 - guidance_gate_init)
        )
        self.guidance_gate_logit = nn.Parameter(
            torch.tensor(guidance_gate_logit)
        )

        self.last_guidance_distribution = None
        self.last_guidance_feature = None

    def forward(self, inp):
        shallow, middle, deep = self.hierarchical_features(inp)

        geometry_feature = torch.cat((
            self.geometry_shallow(shallow),
            self.geometry_middle(middle),
        ), dim=1)
        geometry_logits = self.geometry_head(geometry_feature)
        probabilities = self.geometry_probabilities(geometry_logits)

        # Softmax logits provide deterministic guidance while the Gaussian
        # branch retains its original Gumbel/hard probability behavior.
        guidance_distribution = F.softmax(geometry_logits, dim=1)
        guidance_feature = (
            torch.sigmoid(self.guidance_gate_logit)
            * self.geometry_guidance(guidance_distribution)
        )

        appearance_delta = torch.tanh(self.appearance_fusion(torch.cat((
            shallow,
            middle,
            deep,
            guidance_feature,
        ), dim=1)))
        appearance_residual = (
            torch.sigmoid(self.appearance_gate_logit)
            * appearance_delta
        )
        appearance = deep + appearance_residual

        entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        self.last_geometry_probabilities = probabilities.detach()
        self.last_geometry_entropy = entropy.detach()
        self.last_appearance_residual = appearance_residual.detach()
        self.last_guidance_distribution = guidance_distribution.detach()
        self.last_guidance_feature = guidance_feature.detach()
        return appearance, probabilities


@register('edsr-face-geometry-guided-appearance')
def make_geometry_guided_appearance_encoder(
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
        fusion_dim=64,
        appearance_gate_init=0.25,
        guidance_dim=32,
        guidance_gate_init=0.25):
    if use_pretrained:
        raise ValueError(
            'Geometry-guided encoder requires random initialization.'
        )
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
    return GeometryGuidedAppearanceEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        fusion_dim=fusion_dim,
        appearance_gate_init=appearance_gate_init,
        guidance_dim=guidance_dim,
        guidance_gate_init=guidance_gate_init,
    )
