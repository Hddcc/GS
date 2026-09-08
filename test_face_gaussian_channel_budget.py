import random

import numpy as np
import torch

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_spec(channels):
    return {
        'name': 'gaussian-splatter-memory-efficient',
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
            'primitive_chunk_size': 7,
            'gaussian_channels': channels,
        },
    }


def make_queries(height, width, count):
    coord = utils.make_coord((height, width))
    indices = torch.linspace(0, len(coord) - 1, steps=count).long()
    coord = coord[indices].unsqueeze(0).contiguous()
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


def main():
    seed_all(31)
    inp = torch.rand(1, 3, 6, 9) * 2 - 1
    coord, cell = make_queries(9, 14, 41)
    scale = torch.tensor([1.5])
    parameter_counts = []
    for channels in (16, 32, 64):
        seed_all(31)
        model = models.make(make_spec(channels)).train()
        output = model(inp, coord, scale, cell)
        if output.shape != (1, 41, 3):
            raise RuntimeError('Unexpected output shape for Cg={}.'.format(channels))
        if not torch.isfinite(output).all():
            raise RuntimeError('Non-finite output for Cg={}.'.format(channels))
        output.square().mean().backward()
        active_gradients = sum(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all().item()
            and parameter.grad.abs().sum().item() > 0
            for parameter in model.parameters()
        )
        if active_gradients == 0:
            raise RuntimeError('No active gradients for Cg={}.'.format(channels))
        parameter_counts.append(sum(
            parameter.numel() for parameter in model.parameters()
        ))
        print(
            'Cg={}: output={}, active_gradients={}, layer_elements={}'.format(
                channels, tuple(output.shape), active_gradients,
                model.last_peak_layer_elements,
            )
        )
    if len(set(parameter_counts)) != 1:
        raise RuntimeError('Changing Cg changed the parameter count.')
    print('parameters:', parameter_counts[0])
    print('GAUSSIAN CHANNEL-BUDGET MECHANISM TEST PASSED')


if __name__ == '__main__':
    main()
