import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from models.edsr import EDSR, SegmentationHead


def inverse_softplus(value):
    return math.log(math.expm1(value))


class GeneralizedDivisiveNormalization(nn.Module):
    def __init__(self, channels, beta_min=1e-6, gamma_init=0.1):
        super().__init__()
        self.channels = int(channels)
        self.beta_min = float(beta_min)
        self.beta_reparam = nn.Parameter(torch.full(
            (channels,), inverse_softplus(1.0 - beta_min)
        ))
        gamma = torch.full((channels, channels), -20.0)
        gamma.diagonal().fill_(inverse_softplus(gamma_init))
        self.gamma_reparam = nn.Parameter(gamma)

    def forward(self, x):
        beta = F.softplus(self.beta_reparam) + self.beta_min
        gamma = F.softplus(self.gamma_reparam).view(
            self.channels, self.channels, 1, 1
        )
        normalization = F.conv2d(x.square(), gamma, beta)
        return x * torch.rsqrt(normalization.clamp_min(self.beta_min))


class InformationDensityEstimator(nn.Module):
    def __init__(self, channels, likelihood_floor=1e-9):
        super().__init__()
        self.mean_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.mean_gdn = GeneralizedDivisiveNormalization(channels)
        self.scale_conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.scale_gdn = GeneralizedDivisiveNormalization(channels)
        self.likelihood_floor = float(likelihood_floor)

    def forward(self, feature):
        mean = self.mean_gdn(self.mean_conv(feature))
        scale = F.relu(self.scale_gdn(self.scale_conv(feature))) + 1e-6
        if self.training:
            # Preserve the global RNG stream so the baseline and candidate use
            # identical subsequent Gumbel draws during paired training.
            if feature.is_cuda:
                state = torch.cuda.get_rng_state(feature.device)
                noise = torch.empty_like(feature).uniform_(-0.5, 0.5)
                torch.cuda.set_rng_state(state, feature.device)
            else:
                state = torch.get_rng_state()
                noise = torch.empty_like(feature).uniform_(-0.5, 0.5)
                torch.set_rng_state(state)
            quantized = feature + noise
        else:
            quantized = feature.round()
        inverse_sqrt_two = 1 / math.sqrt(2)
        upper = (quantized + 0.5 - mean) / scale
        lower = (quantized - 0.5 - mean) / scale
        upper_cdf = 0.5 * (1 + torch.erf(upper * inverse_sqrt_two))
        lower_cdf = 0.5 * (1 + torch.erf(lower * inverse_sqrt_two))
        likelihood = (upper_cdf - lower_cdf).clamp_min(
            self.likelihood_floor
        )
        entropy = -torch.log2(likelihood).mean()
        return mean, scale, entropy


class AffineRecalibrationModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.modulation = nn.Conv2d(channels * 2, channels, 1)
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels
        )

    def forward(self, feature, information_density):
        modulation_input, local_input = self.expand(feature).chunk(2, dim=1)
        scale = self.modulation(torch.cat(
            [modulation_input, information_density], dim=1
        ))
        local = self.local(local_input)
        return local * scale


class IAFMIDMARMEncoder(nn.Module):
    def __init__(self, args, n_class=100, arm_positions=(4, 8, 12, 16),
                 arm_residual_scale=0.1, entropy_loss_weight=1e-4):
        super().__init__()
        positions = tuple(int(position) for position in arm_positions)
        if not positions or tuple(sorted(set(positions))) != positions:
            raise ValueError('ARM positions must be sorted and unique.')
        if positions[0] < 1 or positions[-1] > args.n_resblocks:
            raise ValueError('ARM positions must reference EDSR residual blocks.')
        if arm_residual_scale <= 0 or entropy_loss_weight <= 0:
            raise ValueError('IAFM scales and loss weight must be positive.')

        # Construct the unchanged baseline modules first to align shared RNG.
        self.args = args
        self.edsr = EDSR(args)
        self.segmentation_head = SegmentationHead(
            in_channels=args.n_feats, out_channels=n_class
        )
        self.out_dim = args.n_feats

        shared_rng_state = torch.get_rng_state()
        self.information_estimator = InformationDensityEstimator(args.n_feats)
        self.arm_positions = positions
        self.arm_modules = nn.ModuleDict({
            str(position): AffineRecalibrationModule(args.n_feats)
            for position in positions
        })
        torch.set_rng_state(shared_rng_state)
        self.arm_residual_scale = float(arm_residual_scale)
        self.entropy_loss_weight = float(entropy_loss_weight)
        self.last_information_entropy_loss = None
        self.last_iafm_diagnostics = None

    def training_diagnostics(self):
        if self.last_information_entropy_loss is None:
            raise RuntimeError('IAFM diagnostics requested before forward.')
        return {
            'information_entropy_loss': self.last_information_entropy_loss,
            'idm_mean': self.last_iafm_diagnostics['idm_mean'],
            'idm_spatial_std': self.last_iafm_diagnostics['idm_spatial_std'],
            'arm_response': self.last_iafm_diagnostics['arm_response'],
            'scaled_arm_response': self.last_iafm_diagnostics[
                'scaled_arm_response'
            ],
        }

    def forward(self, x):
        shallow = self.edsr.head(x)
        _, information_density, entropy = self.information_estimator(shallow)
        feature = shallow
        arm_responses = []
        for index, block in enumerate(self.edsr.body):
            feature = block(feature)
            position = index + 1
            if position in self.arm_positions:
                response = self.arm_modules[str(position)](
                    feature, information_density
                )
                feature = feature + self.arm_residual_scale * response
                arm_responses.append(response.abs().mean())
        feature = feature + shallow

        logits = self.segmentation_head(feature)
        batch, classes, height, width = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(
            batch * height * width, classes
        )
        if self.training:
            logits = F.gumbel_softmax(logits, tau=1, hard=False)
        else:
            indices = logits.argmax(dim=-1, keepdim=True)
            logits = torch.zeros_like(logits).scatter_(1, indices, 1.0)
        logits = logits.view(batch, height, width, classes).permute(
            0, 3, 1, 2
        ).contiguous()

        responses = torch.stack(arm_responses)
        spatial_std = information_density.flatten(2).std(dim=-1).mean()
        self.last_information_entropy_loss = entropy
        self.last_iafm_diagnostics = {
            'idm_mean': information_density.mean().detach(),
            'idm_spatial_std': spatial_std.detach(),
            'arm_response': responses.detach(),
            'scaled_arm_response': (
                responses * self.arm_residual_scale
            ).detach(),
        }
        return feature, logits


@register('edsr-face-iafm-idm-arm')
def make_iafm_idm_arm_encoder(
        n_resblocks=16, n_feats=64, res_scale=1, scale=2,
        no_upsampling=True, rgb_range=1, n_class=100,
        use_pretrained=False, arm_positions=(4, 8, 12, 16),
        arm_residual_scale=0.1, entropy_loss_weight=1e-4):
    if use_pretrained:
        raise ValueError('IAFM IDM+ARM encoder requires random initialization.')
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale
    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.rgb_range = rgb_range
    args.n_colors = 3
    args.pretrained_path = None
    return IAFMIDMARMEncoder(
        args,
        n_class=n_class,
        arm_positions=arm_positions,
        arm_residual_scale=arm_residual_scale,
        entropy_loss_weight=entropy_loss_weight,
    )
