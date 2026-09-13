import argparse
import random

import numpy as np
import torch

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_spec():
    return {
        'name': 'gaussian-splatter',
        'args': {
            'encoder_spec': {
                'name': 'edsr-baseline',
                'args': {
                    'n_resblocks': 2,
                    'n_feats': 64,
                    'no_upsampling': True,
                    'use_pretrained': False,
                },
            },
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

    seed_all(71)
    model = models.make(make_spec()).to(device).eval()
    inp = torch.rand(1, 3, 6, 6, device=device) * 2 - 1
    coord, cell = make_query(9, 9, 41, device)
    scale = torch.tensor([1.5], device=device)
    with torch.no_grad():
        model.set_oracle_covariance_adjustment(None)
        baseline = model(inp, coord, scale, cell)
        model.set_oracle_covariance_adjustment(None)
        repeated = model(inp, coord, scale, cell)
        if not torch.equal(baseline, repeated):
            raise RuntimeError('Inactive oracle hook changed baseline output.')

        adjustment = torch.tensor(
            [[0.2, 0.05], [0.05, 0.1]], device=device
        )
        model.set_oracle_covariance_adjustment(adjustment, 'add')
        added = model(inp, coord, scale, cell)
        add_diagnostic = model.last_oracle_covariance_diagnostics
        model.set_oracle_covariance_adjustment(
            adjustment * 100, 'subtract', min_eigenvalue=1e-4
        )
        subtracted = model(inp, coord, scale, cell)
        subtract_diagnostic = model.last_oracle_covariance_diagnostics

    for label, output in (('add', added), ('subtract', subtracted)):
        if output.shape != baseline.shape or not torch.isfinite(output).all():
            raise RuntimeError('{} oracle output is invalid.'.format(label))
        if torch.equal(output, baseline):
            raise RuntimeError('{} oracle adjustment is inactive.'.format(label))
    if add_diagnostic['minimum_eigenvalue'].item() <= 0:
        raise RuntimeError('Positive oracle covariance is not SPD.')
    if subtract_diagnostic['clamp_fraction'].item() <= 0:
        raise RuntimeError('PSD-safe subtraction did not exercise its clamp.')
    if subtract_diagnostic['minimum_eigenvalue'].item() < 0.999e-4:
        raise RuntimeError('PSD-safe subtraction violated the eigenvalue floor.')

    print('device:', device)
    print('baseline/add/subtract responses: {:.8f}/{:.8f}/{:.8f}'.format(
        baseline.abs().mean().item(),
        added.abs().mean().item(),
        subtracted.abs().mean().item(),
    ))
    print('subtraction clamp fraction: {:.8f}'.format(
        subtract_diagnostic['clamp_fraction'].item()))
    print('BLIND-SPD ORACLE SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
