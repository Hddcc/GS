import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

import models
import utils
from models.edsr_face_iafm_idm_arm import (
    AffineRecalibrationModule,
    GeneralizedDivisiveNormalization,
    InformationDensityEstimator,
)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_spec(candidate):
    encoder_args = {
        'n_resblocks': 4,
        'n_feats': 64,
        'no_upsampling': True,
        'use_pretrained': False,
    }
    encoder_name = 'edsr-baseline'
    if candidate:
        encoder_name = 'edsr-face-iafm-idm-arm'
        encoder_args.update({
            'arm_positions': [1, 2, 3, 4],
            'arm_residual_scale': 0.1,
            'entropy_loss_weight': 1e-4,
        })
    return {
        'name': 'gaussian-splatter',
        'args': {
            'encoder_spec': {'name': encoder_name, 'args': encoder_args},
            'dec_spec': {
                'name': 'mlp',
                'args': {'out_dim': 3, 'hidden_list': [32, 32]},
            },
            'kernel_size': 5,
            'hidden_dim': 32,
        },
    }


def make_query(height, width, count, device):
    coord = utils.make_coord((height, width))
    indices = torch.linspace(0, len(coord) - 1, steps=count).long()
    coord = coord[indices].unsqueeze(0).to(device)
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


def shared_initialization_error(baseline, candidate):
    candidate_state = candidate.state_dict()
    errors = []
    compared = 0
    for name, value in baseline.state_dict().items():
        if name in candidate_state and candidate_state[name].shape == value.shape:
            errors.append((candidate_state[name] - value).abs().max().item())
            compared += value.numel()
    return max(errors), compared


def active_gradient(parameter, label):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError('{} has no finite gradient.'.format(label))
    value = parameter.grad.abs().mean().item()
    if value <= 0:
        raise RuntimeError('{} gradient is zero.'.format(label))
    return value


def check_equations(device):
    gdn = GeneralizedDivisiveNormalization(3).to(device)
    feature = torch.randn(2, 3, 5, 7, device=device)
    output = gdn(feature)
    beta = F.softplus(gdn.beta_reparam) + gdn.beta_min
    gamma = F.softplus(gdn.gamma_reparam).view(3, 3, 1, 1)
    reference = feature * torch.rsqrt(
        F.conv2d(feature.square(), gamma, beta).clamp_min(gdn.beta_min)
    )
    gdn_error = (output - reference).abs().max().item()
    if gdn_error > 1e-7:
        raise RuntimeError('GDN reference equation failed.')

    arm = AffineRecalibrationModule(4).to(device)
    feature = torch.randn(2, 4, 5, 7, device=device)
    density = torch.rand_like(feature)
    expanded = arm.expand(feature)
    modulation_input, local_input = expanded.chunk(2, dim=1)
    reference = arm.local(local_input) * arm.modulation(torch.cat(
        [modulation_input, density], dim=1
    ))
    arm_error = (arm(feature, density) - reference).abs().max().item()
    if arm_error > 1e-7:
        raise RuntimeError('ARM reference equation failed.')

    estimator = InformationDensityEstimator(4).to(device).train()
    feature = torch.randn(2, 4, 5, 7, device=device)
    if device.type == 'cuda':
        before = torch.cuda.get_rng_state(device)
    else:
        before = torch.get_rng_state()
    _, density, entropy = estimator(feature)
    if device.type == 'cuda':
        after = torch.cuda.get_rng_state(device)
    else:
        after = torch.get_rng_state()
    if not torch.equal(before, after):
        raise RuntimeError('IDM quantization changed the shared RNG stream.')
    if not torch.isfinite(entropy) or density.min().item() <= 0:
        raise RuntimeError('IDM estimator output is invalid.')
    return gdn_error, arm_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable.')
        device = torch.device('cuda:{}'.format(args.gpu))
    else:
        device = torch.device('cpu')
    seed_all(83)
    gdn_error, arm_error = check_equations(device)

    seed_all(89)
    baseline = models.make(make_spec(False)).to(device)
    seed_all(89)
    candidate = models.make(make_spec(True)).to(device)
    initialization_error, compared = shared_initialization_error(
        baseline, candidate
    )
    if initialization_error != 0:
        raise RuntimeError('Shared initialization changed.')
    parameter_delta = (
        sum(parameter.numel() for parameter in candidate.parameters())
        - sum(parameter.numel() for parameter in baseline.parameters())
    )
    if parameter_delta != 151040:
        raise RuntimeError('Unexpected IAFM parameter delta.')

    inp = torch.rand(1, 3, 6, 6, device=device) * 2 - 1
    coord, cell = make_query(9, 9, 41, device)
    scale = torch.tensor([1.5], device=device)
    candidate.train()
    prediction, diagnostics = candidate(inp, coord, scale, cell)
    entropy = diagnostics['information_entropy_loss'].mean()
    loss = prediction.square().mean() + 1e-4 * entropy
    loss.backward()
    gradients = {
        'mean': active_gradient(
            candidate.encoder.information_estimator.mean_conv.weight, 'mean'
        ),
        'scale': active_gradient(
            candidate.encoder.information_estimator.scale_conv.weight, 'scale'
        ),
        'arm': active_gradient(
            candidate.encoder.arm_modules['1'].modulation.weight, 'ARM'
        ),
        'encoder': active_gradient(
            candidate.encoder.edsr.head[0].weight, 'encoder'
        ),
    }
    if diagnostics['idm_spatial_std'].mean().item() <= 1e-4:
        raise RuntimeError('IDM has collapsed spatially.')
    if diagnostics['scaled_arm_response'].min().item() <= 1e-4:
        raise RuntimeError('An ARM insertion is inactive.')

    candidate.eval()
    with torch.no_grad():
        first = candidate(inp, coord, scale, cell)
        second = candidate(inp, coord, scale, cell)
    if not torch.equal(first, second):
        raise RuntimeError('IAFM evaluation is not deterministic.')
    if first.shape != (1, 41, 3) or not torch.isfinite(first).all():
        raise RuntimeError('IAFM evaluation output is invalid.')

    print('device:', device)
    print('parameter delta:', parameter_delta)
    print('shared parameters compared:', compared)
    print('shared initialization error: {:.2e}'.format(initialization_error))
    print('GDN/ARM equation errors: {:.2e}/{:.2e}'.format(
        gdn_error, arm_error))
    print('entropy/idm std: {:.6f}/{:.6f}'.format(
        entropy.item(), diagnostics['idm_spatial_std'].mean().item()))
    print('scaled ARM responses:', '/'.join(
        '{:.6f}'.format(value)
        for value in diagnostics['scaled_arm_response'].tolist()
    ))
    print('gradients:', '/'.join(
        '{}={:.3e}'.format(name, value) for name, value in gradients.items()
    ))
    print('IAFM IDM+ARM MECHANISM TEST PASSED')


if __name__ == '__main__':
    main()
