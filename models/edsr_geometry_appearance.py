import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.edsr import EDSR


class GeometryAppearanceEncoder(nn.Module):
    """Assign shallow structure and deep appearance to separate GS heads."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            fusion_dim=64,
            appearance_gate_init=0.25):
        super().__init__()
        if not 0 < shallow_depth < middle_depth <= args.n_resblocks:
            raise ValueError(
                'Expected 0 < shallow_depth < middle_depth <= n_resblocks.'
            )
        if not 0 < appearance_gate_init < 1:
            raise ValueError('appearance_gate_init must be between 0 and 1.')

        self.args = args
        self.edsr = EDSR(args)
        self.out_dim = args.n_feats
        self.shallow_depth = shallow_depth
        self.middle_depth = middle_depth

        self.geometry_shallow = nn.Sequential(
            nn.Conv2d(args.n_feats, geometry_dim, 1),
            nn.SiLU(inplace=True),
        )
        self.geometry_middle = nn.Sequential(
            nn.Conv2d(args.n_feats, geometry_dim, 1),
            nn.SiLU(inplace=True),
        )
        self.geometry_head = nn.Sequential(
            nn.Conv2d(2 * geometry_dim, args.n_feats, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(args.n_feats, n_class, 3, padding=1),
        )

        self.appearance_fusion = nn.Sequential(
            nn.Conv2d(3 * args.n_feats, fusion_dim, 1),
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
        gate_logit = math.log(
            appearance_gate_init / (1 - appearance_gate_init)
        )
        self.appearance_gate_logit = nn.Parameter(torch.tensor(gate_logit))

        self.last_geometry_probabilities = None
        self.last_geometry_entropy = None
        self.last_appearance_residual = None

    def hierarchical_features(self, inp):
        head = self.edsr.head(inp)
        feature = head
        shallow = None
        middle = None
        residual_blocks = self.edsr.body[:-1]
        for index, block in enumerate(residual_blocks, start=1):
            feature = block(feature)
            if index == self.shallow_depth:
                shallow = feature
            if index == self.middle_depth:
                middle = feature
        deep = self.edsr.body[-1](feature) + head
        if shallow is None or middle is None:
            raise RuntimeError('Failed to capture hierarchical EDSR features.')
        return shallow, middle, deep

    def geometry_probabilities(self, logits):
        batch, classes, height, width = logits.shape
        flat = logits.permute(0, 2, 3, 1).reshape(-1, classes)
        if self.training:
            probabilities = F.gumbel_softmax(flat, tau=1, hard=False)
        else:
            indices = flat.argmax(dim=-1, keepdim=True)
            probabilities = torch.zeros_like(flat).scatter_(1, indices, 1.0)
        return probabilities.reshape(
            batch, height, width, classes
        ).permute(0, 3, 1, 2).contiguous()

    def forward(self, inp):
        shallow, middle, deep = self.hierarchical_features(inp)

        geometry_feature = torch.cat((
            self.geometry_shallow(shallow),
            self.geometry_middle(middle),
        ), dim=1)
        geometry_logits = self.geometry_head(geometry_feature)
        probabilities = self.geometry_probabilities(geometry_logits)

        appearance_delta = torch.tanh(self.appearance_fusion(torch.cat((
            shallow,
            middle,
            deep,
        ), dim=1)))
        appearance_residual = (
            torch.sigmoid(self.appearance_gate_logit) * appearance_delta
        )
        appearance = deep + appearance_residual

        entropy = -(
            probabilities.clamp_min(1e-8)
            * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        self.last_geometry_probabilities = probabilities.detach()
        self.last_geometry_entropy = entropy.detach()
        self.last_appearance_residual = appearance_residual.detach()
        return appearance, probabilities


@register('edsr-face-geometry-appearance')
def make_geometry_appearance_encoder(
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
        appearance_gate_init=0.25):
    if use_pretrained:
        raise ValueError(
            'Geometry-appearance encoder requires random initialization.'
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
    return GeometryAppearanceEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        fusion_dim=fusion_dim,
        appearance_gate_init=appearance_gate_init,
    )
