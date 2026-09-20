import argparse
import math
import os
import random

import numpy as np
import torch
import yaml

import models
import utils


def make_queries(inp_size, scale, sample_q, device):
    height = round(inp_size * scale)
    width = round(inp_size * scale)
    coord = utils.make_coord((height, width)).to(device)
    if coord.shape[0] > sample_q:
        indices = torch.linspace(
            0, coord.shape[0] - 1, sample_q, device=device
        ).round().long()
        coord = coord[indices]
    coord = coord.unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / height
    cell[:, :, 1] *= 2 / width
    return coord, cell


def assert_finite(name, value):
    if not torch.isfinite(value).all():
        raise RuntimeError('{} contains NaN or Inf.'.format(name))


def gradient_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            assert_finite('frequency residual gradient', parameter.grad)
            squared_norm += parameter.grad.detach().pow(2).sum().item()
    return math.sqrt(squared_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.5, 4, 8])
    parser.add_argument('--inp-size', type=int, default=16)
    parser.add_argument('--sample-q', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the GaussianSR smoke test.')

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is None:
        raise ValueError('A seed is required in YAML or via --seed.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = models.make(config['model']).cuda().train()
    print('model:', config['model']['name'])
    print('seed:', seed)
    print('parameters:', sum(p.numel() for p in model.parameters()))
    print(
        'initial residual gate:',
        torch.sigmoid(model.residual_gate_logit).detach().item(),
    )

    residual_parameters = list(model.frequency_encoder.parameters())
    residual_parameters += list(model.residual_head.parameters())
    residual_parameters.append(model.residual_gate_logit)

    for scale_value in args.scales:
        model.zero_grad(set_to_none=True)
        inp = torch.rand(
            1, 3, args.inp_size, args.inp_size, device='cuda'
        ) * 2 - 1
        coord, cell = make_queries(
            args.inp_size, scale_value, args.sample_q, inp.device
        )
        scale = torch.tensor(
            [scale_value], dtype=torch.float32, device=inp.device
        )
        pred = model(inp, coord, scale, cell)
        assert_finite('prediction', pred)
        assert_finite('RGB residual', model.last_rgb_residual)

        loss = pred.abs().mean()
        loss.backward()
        assert_finite('loss', loss)
        grad_norm = gradient_norm(residual_parameters)
        if grad_norm == 0:
            raise RuntimeError('Frequency residual gradient is zero.')

        residual = model.last_rgb_residual
        print(
            'scale={:g}, pred={}, loss={:.6f}, residual_abs_mean={:.8f}, '
            'residual_abs_max={:.8f}, residual_grad={:.8f}'.format(
                scale_value,
                tuple(pred.shape),
                loss.item(),
                residual.abs().mean().item(),
                residual.abs().max().item(),
                grad_norm,
            )
        )

    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
