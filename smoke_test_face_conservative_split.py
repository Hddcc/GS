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


def gradient_norm(parameters, name):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            assert_finite('{} gradient'.format(name), parameter.grad)
            squared_norm += parameter.grad.detach().square().sum().item()
    norm = math.sqrt(squared_norm)
    if norm == 0:
        raise RuntimeError('{} gradient is zero.'.format(name))
    return norm


def choose_device(requested):
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.5, 4, 8])
    parser.add_argument('--inp-size', type=int, default=None)
    parser.add_argument('--sample-q', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = choose_device(args.device)
    inp_size = args.inp_size or (16 if device.type == 'cuda' else 4)
    sample_q = args.sample_q or (512 if device.type == 'cuda' else 16)

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is None:
        raise ValueError('A seed is required in YAML or via --seed.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = models.make(config['model']).to(device).train()
    print('model:', config['model']['name'])
    print('device:', device)
    print('seed:', seed)
    print('parameters:', sum(parameter.numel() for parameter in model.parameters()))
    print('input/query:', inp_size, sample_q)

    for scale_value in args.scales:
        model.zero_grad(set_to_none=True)
        inp = torch.rand(1, 3, inp_size, inp_size, device=device) * 2 - 1
        coord, cell = make_queries(inp_size, scale_value, sample_q, device)
        scale = torch.tensor([scale_value], dtype=torch.float32, device=device)
        pred = model(inp, coord, scale, cell)
        assert_finite('prediction', pred)
        assert_finite('split offset', model.last_split_offset)
        assert_finite('split blend', model.last_split_blend)
        assert_finite('split feature delta', model.last_split_feature_delta)

        center_error = model.last_center_conservation_error.item()
        feature_error = model.last_feature_conservation_error.item()
        if center_error > 1e-7 or feature_error > 1e-7:
            raise RuntimeError('Conservative split invariant was violated.')
        blend_min = model.last_split_blend.min().item()
        blend_max = model.last_split_blend.max().item()
        if blend_min <= 0 or blend_max >= 1:
            raise RuntimeError('Split blend must remain between zero and one.')

        loss = pred.abs().mean()
        loss.backward()
        predictor_grad = gradient_norm(
            model.split_predictor.parameters(), 'split predictor'
        )
        gate_grad = gradient_norm([model.split_gate_logit], 'split gate')
        if model.last_split_feature_delta.abs().mean().item() <= 1e-5:
            raise RuntimeError('Paired feature residual is effectively dormant.')
        if predictor_grad < 1e-8 or gate_grad < 1e-9:
            raise RuntimeError(
                'Conservative split gradient is too weak for a training probe.'
            )
        print(
            'scale={:g}, pred={}, loss={:.6f}, offset={:.8f}, '
            'blend={:.8f}, feature_delta={:.8f}, '
            'conservation=({:.2e},{:.2e}), grads=split:{:.3e}/gate:{:.3e}'
            .format(
                scale_value,
                tuple(pred.shape),
                loss.item(),
                model.last_split_offset.norm(dim=-1).mean().item(),
                model.last_split_blend.mean().item(),
                model.last_split_feature_delta.abs().mean().item(),
                center_error,
                feature_error,
                predictor_grad,
                gate_grad,
            )
        )

    model.eval()
    with torch.no_grad():
        inp = torch.rand(1, 3, inp_size, inp_size, device=device) * 2 - 1
        coord, cell = make_queries(inp_size, 4, sample_q, device)
        scale = torch.tensor([4.0], dtype=torch.float32, device=device)
        output = model(inp, coord, scale, cell)
    if not torch.is_tensor(output):
        raise RuntimeError('Evaluation forward must return a prediction tensor.')
    assert_finite('evaluation prediction', output)
    print('evaluation output:', tuple(output.shape))
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
