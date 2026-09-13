import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_spec():
    return {
        'name': 'gaussian-splatter',
        'args': {
            'encoder_spec': {
                'name': 'edsr-face-iafm-idm-arm',
                'args': {
                    'n_resblocks': 4,
                    'n_feats': 64,
                    'no_upsampling': True,
                    'use_pretrained': False,
                    'arm_positions': [1, 2, 3, 4],
                    'arm_residual_scale': 0.1,
                    'entropy_loss_weight': 1e-4,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical-gpus', default='0,2,4')
    args = parser.parse_args()
    expected = args.physical_gpus.split(',')
    if len(expected) != 3 or len(set(expected)) != 3:
        raise ValueError('Exactly three distinct physical GPUs are required.')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible != args.physical_gpus:
        raise RuntimeError('CUDA_VISIBLE_DEVICES does not match physical-gpus.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError('Three visible CUDA devices are required.')

    seed_all(97)
    model = nn.DataParallel(models.make(make_spec()).cuda()).train()
    batch_size = 12
    inp = torch.rand(batch_size, 3, 6, 6, device='cuda') * 2 - 1
    coord = utils.make_coord((9, 9))
    indices = torch.linspace(0, len(coord) - 1, steps=41).long()
    coord = coord[indices].unsqueeze(0).expand(batch_size, -1, -1).cuda()
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / 9
    cell[..., 1] *= 2 / 9
    scale = torch.full((batch_size,), 1.5, device='cuda')
    prediction, diagnostics = model(inp, coord, scale, cell)
    if prediction.shape != (batch_size, 41, 3):
        raise RuntimeError('DataParallel prediction shape is incorrect.')
    if diagnostics['information_entropy_loss'].shape != (3,):
        raise RuntimeError('DataParallel entropy gather shape is incorrect.')
    if diagnostics['arm_response'].numel() != 12:
        raise RuntimeError('DataParallel ARM diagnostics were not gathered.')
    loss = (
        prediction.square().mean()
        + 1e-4 * diagnostics['information_entropy_loss'].mean()
    )
    loss.backward()
    gradient = model.module.encoder.information_estimator.scale_conv.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() \
            or gradient.abs().mean().item() <= 0:
        raise RuntimeError('DataParallel IAFM gradient is invalid.')
    print('physical GPUs:', args.physical_gpus)
    print('prediction:', tuple(prediction.shape))
    print('entropy gather:', tuple(diagnostics['information_entropy_loss'].shape))
    print('ARM diagnostic elements:', diagnostics['arm_response'].numel())
    print('IAFM IDM+ARM DATAPARALLEL SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
