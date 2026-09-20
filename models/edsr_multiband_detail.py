from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.edsr_geometry_appearance import GeometryAppearanceEncoder


class MultiBandDetailEncoder(GeometryAppearanceEncoder):
    """Add fixed multi-band image detail to appearance restoration."""

    def __init__(
            self,
            args,
            n_class=100,
            shallow_depth=4,
            middle_depth=10,
            geometry_dim=32,
            fusion_dim=64,
            appearance_gate_init=0.25,
            detail_dim=32):
        super().__init__(
            args=args,
            n_class=n_class,
            shallow_depth=shallow_depth,
            middle_depth=middle_depth,
            geometry_dim=geometry_dim,
            fusion_dim=fusion_dim,
            appearance_gate_init=appearance_gate_init,
        )
        if detail_dim <= 0:
            raise ValueError('detail_dim must be positive.')

        kernel_3 = torch.tensor([1.0, 2.0, 1.0])
        kernel_3 = torch.outer(kernel_3, kernel_3)
        kernel_3 = kernel_3 / kernel_3.sum()
        kernel_5 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel_5 = torch.outer(kernel_5, kernel_5)
        kernel_5 = kernel_5 / kernel_5.sum()
        self.register_buffer('gaussian_kernel_3', kernel_3[None, None])
        self.register_buffer('gaussian_kernel_5', kernel_5[None, None])

        self.detail_encoder = nn.Sequential(
            nn.Conv2d(6, detail_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_dim, args.n_feats, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.appearance_fusion = nn.Sequential(
            nn.Conv2d(4 * args.n_feats, fusion_dim, 1),
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

        self.last_high_band = None
        self.last_middle_band = None
        self.last_detail_feature = None

    def gaussian_blur(self, inp, kernel):
        channels = inp.shape[1]
        padding = kernel.shape[-1] // 2
        weight = kernel.to(dtype=inp.dtype).expand(channels, 1, -1, -1)
        padded = F.pad(
            inp,
            (padding, padding, padding, padding),
            mode='replicate',
        )
        return F.conv2d(padded, weight, groups=channels)

    def detail_bands(self, inp):
        blur_3 = self.gaussian_blur(inp, self.gaussian_kernel_3)
        blur_5 = self.gaussian_blur(inp, self.gaussian_kernel_5)
        high_band = inp - blur_3
        middle_band = blur_3 - blur_5
        return high_band, middle_band

    def forward(self, inp):
        shallow, middle, deep = self.hierarchical_features(inp)

        geometry_feature = torch.cat((
            self.geometry_shallow(shallow),
            self.geometry_middle(middle),
        ), dim=1)
        geometry_logits = self.geometry_head(geometry_feature)
        probabilities = self.geometry_probabilities(geometry_logits)

        high_band, middle_band = self.detail_bands(inp)
        detail_feature = self.detail_encoder(torch.cat((
            high_band,
            middle_band,
        ), dim=1))
        appearance_delta = torch.tanh(self.appearance_fusion(torch.cat((
            shallow,
            middle,
            deep,
            detail_feature,
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
        self.last_high_band = high_band.detach()
        self.last_middle_band = middle_band.detach()
        self.last_detail_feature = detail_feature.detach()
        return appearance, probabilities


@register('edsr-face-multiband-detail')
def make_multiband_detail_encoder(
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
        detail_dim=32):
    if use_pretrained:
        raise ValueError('Multi-band detail encoder requires random initialization.')
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
    return MultiBandDetailEncoder(
        args=args,
        n_class=n_class,
        shallow_depth=shallow_depth,
        middle_depth=middle_depth,
        geometry_dim=geometry_dim,
        fusion_dim=fusion_dim,
        appearance_gate_init=appearance_gate_init,
        detail_dim=detail_dim,
    )
