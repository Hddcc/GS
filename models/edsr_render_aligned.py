import math
from argparse import Namespace

import torch
import torch.nn as nn

from models import register
from models.edsr_geometry_appearance import GeometryAppearanceEncoder


class RenderAlignedEncoder(GeometryAppearanceEncoder):
    """Align encoder channel duties with Gaussian and bypass rendering."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            fusion_dim=64,
            appearance_gate_init=0.25,
            render_channels=8,
            render_gate_init=0.25):
        super().__init__(
            args=args,
            n_class=n_class,
            shallow_depth=shallow_depth,
            middle_depth=middle_depth,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            appearance_gate_init=appearance_gate_init,
        )
        if render_channels != 8:
            raise ValueError(
                'GaussianSplatter requires exactly 8 render channels.'
            )
        if args.n_feats <= render_channels:
            raise ValueError('n_feats must exceed render_channels.')
        if not 0 < render_gate_init < 1:
            raise ValueError('render_gate_init must be between 0 and 1.')

        bypass_channels = args.n_feats - render_channels
        self.render_channels = render_channels
        self.bypass_channels = bypass_channels
        self.render_fusion = nn.Sequential(
            nn.Conv2d(
                2 * geometry_dim + render_channels,
                fusion_dim,
                3,
                padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_dim, render_channels, 3, padding=1),
        )
        self.appearance_fusion = nn.Sequential(
            nn.Conv2d(3 * args.n_feats, fusion_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(fusion_dim, bypass_channels, 3, padding=1),
        )
        nn.init.normal_(
            self.render_fusion[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.render_fusion[-1].bias)
        nn.init.normal_(
            self.appearance_fusion[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.appearance_fusion[-1].bias)

        render_gate_logit = math.log(
            render_gate_init / (1 - render_gate_init)
        )
        self.render_gate_logit = nn.Parameter(torch.tensor(render_gate_logit))

        self.last_render_residual = None
        self.last_bypass_residual = None
        self.last_render_feature = None
        self.last_bypass_feature = None

    def forward(self, inp):
        shallow, middle, deep = self.hierarchical_features(inp)

        geometry_feature = torch.cat((
            self.geometry_shallow(shallow),
            self.geometry_middle(middle),
        ), dim=1)
        geometry_logits = self.geometry_head(geometry_feature)
        probabilities = self.geometry_probabilities(geometry_logits)

        deep_render = deep[:, :self.render_channels]
        deep_bypass = deep[:, self.render_channels:]
        render_delta = torch.tanh(self.render_fusion(torch.cat((
            geometry_feature,
            deep_render,
        ), dim=1)))
        bypass_delta = torch.tanh(self.appearance_fusion(torch.cat((
            shallow,
            middle,
            deep,
        ), dim=1)))
        render_residual = (
            torch.sigmoid(self.render_gate_logit) * render_delta
        )
        bypass_residual = (
            torch.sigmoid(self.appearance_gate_logit) * bypass_delta
        )
        render_feature = deep_render + render_residual
        bypass_feature = deep_bypass + bypass_residual
        feature = torch.cat((render_feature, bypass_feature), dim=1)

        entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        self.last_geometry_probabilities = probabilities.detach()
        self.last_geometry_entropy = entropy.detach()
        self.last_appearance_residual = bypass_residual.detach()
        self.last_render_residual = render_residual.detach()
        self.last_bypass_residual = bypass_residual.detach()
        self.last_render_feature = render_feature.detach()
        self.last_bypass_feature = bypass_feature.detach()
        return feature, probabilities


@register('edsr-face-render-aligned')
def make_render_aligned_encoder(
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
        render_channels=8,
        render_gate_init=0.25):
    if use_pretrained:
        raise ValueError('Render-aligned encoder requires random initialization.')
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
    return RenderAlignedEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        fusion_dim=fusion_dim,
        appearance_gate_init=appearance_gate_init,
        render_channels=render_channels,
        render_gate_init=render_gate_init,
    )
