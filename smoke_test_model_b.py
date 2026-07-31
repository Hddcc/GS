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

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is None:
        raise ValueError('A seed is required in YAML or via --seed.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = models.make(config['model']).cuda().train()
    parameter_count = sum(p.numel() for p in model.parameters())
    print('model:', config['model']['name'])
    print('seed:', seed)
    print('parameters:', parameter_count)
    if hasattr(model, 'residual_gate_logits'):
        gates = torch.sigmoid(model.residual_gate_logits).detach().cpu().tolist()
        print('initial residual gates:', gates)

    for scale_value in args.scales:
        model.zero_grad(set_to_none=True)
        inp = torch.rand(1, 3, args.inp_size, args.inp_size, device='cuda') * 2 - 1
        coord, cell = make_queries(
            args.inp_size, scale_value, args.sample_q, inp.device
        )
        scale = torch.tensor([scale_value], dtype=torch.float32, device=inp.device)
        pred = model(inp, coord, scale, cell)
        assert_finite('prediction', pred)
        if model.last_gaussian_deltas is not None:
            max_delta = max(
                value.abs().max().item()
                for value in model.last_gaussian_deltas.values()
            )
            if max_delta > 1e-7:
                raise RuntimeError(
                    'Zero-initialized Gaussian residuals are not zero: {:.8f}'
                    .format(max_delta)
                )

        loss = pred.abs().mean()
        loss.backward()
        assert_finite('loss', loss)

        grad_norm = 0.0
        for parameter in model.local_modulator.parameters():
            if parameter.grad is not None:
                assert_finite('local modulator gradient', parameter.grad)
                grad_norm += parameter.grad.detach().pow(2).sum().item()
        grad_norm = math.sqrt(grad_norm)
        if grad_norm == 0:
            raise RuntimeError('Local modulator gradient is zero.')

        names = ('sigma_x', 'sigma_y', 'rho', 'opacity')
        stats = []
        for name, value in zip(names, model.last_gaussian_parameters):
            assert_finite(name, value)
            stats.append(
                '{}=[min={:.4f}, max={:.4f}, std={:.4f}]'.format(
                    name,
                    value.min().item(),
                    value.max().item(),
                    value.std().item(),
                )
            )
        print(
            'scale={:g}, pred={}, loss={:.6f}, modulator_grad={:.6f}'.format(
                scale_value, tuple(pred.shape), loss.item(), grad_norm
            )
        )
        print('  ' + ', '.join(stats))

    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
