import argparse
import copy
import os
import random

import numpy as np
import torch
import yaml

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_queries(height, width, count, batch_size, device):
    coord = utils.make_coord((height, width)).to(device)
    indices = torch.linspace(
        0, len(coord) - 1, steps=count, device=device
    ).long()
    coord = coord[indices].unsqueeze(0).expand(
        batch_size, -1, -1
    ).contiguous()
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


def baseline_spec(candidate_spec):
    spec = copy.deepcopy(candidate_spec)
    spec['name'] = 'gaussian-splatter-memory-efficient'
    for key in (
        'primitive_count', 'residual_limit', 'offset_residual_limit',
        'covariance_log_limit',
    ):
        spec['args'].pop(key)
    return spec


def shared_initialization_error(baseline, candidate):
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    errors = []
    compared = 0
    for name, value in baseline_state.items():
        if name in candidate_state and candidate_state[name].shape == value.shape:
            errors.append((candidate_state[name] - value).abs().max().item())
            compared += value.numel()
    return max(errors), compared


def gradient_mean(parameter, label):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError('{} has no finite gradient.'.format(label))
    value = parameter.grad.abs().mean().item()
    if value <= 0:
        raise RuntimeError('{} gradient is zero.'.format(label))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--inp-size', type=int, default=16)
    parser.add_argument('--sample-q', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device(args.device)
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    candidate_spec = config['model']
    seed_all(args.seed)
    baseline = models.make(baseline_spec(candidate_spec)).to(device)
    seed_all(args.seed)
    candidate = models.make(candidate_spec).to(device)
    initialization_error, compared = shared_initialization_error(
        baseline, candidate
    )
    if initialization_error != 0:
        raise RuntimeError('Shared initialization changed.')
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    if candidate_parameters - baseline_parameters != 2100:
        raise RuntimeError('Unexpected parameter delta.')

    batch_size = args.batch_size
    inp = torch.rand(
        batch_size, 3, args.inp_size, args.inp_size + 2, device=device
    ) * 2 - 1
    print('model:', candidate_spec['name'])
    print('device:', args.device)
    print('seed:', args.seed)
    print('parameters: baseline={}, candidate={}, delta={}'.format(
        baseline_parameters,
        candidate_parameters,
        candidate_parameters - baseline_parameters,
    ))
    print('shared parameters compared:', compared)
    print('shared initialization error: {:.2e}'.format(initialization_error))
    print('initial offsets:', candidate.primitive_offsets().detach().cpu().tolist())
    print('initial opacity weights:', candidate.opacity_weights().detach().cpu().tolist())

    baseline.eval()
    candidate.eval()
    with torch.no_grad():
        for scale_value in (1.5, 4.0, 8.0):
            height = round(args.inp_size * scale_value)
            width = round((args.inp_size + 2) * scale_value)
            coord, cell = make_queries(
                height, width, args.sample_q, batch_size, device
            )
            scale = torch.full((batch_size,), scale_value, device=device)
            prediction = candidate(inp, coord, scale, cell)
            baseline_prediction = baseline(inp, coord, scale, cell)
            if prediction.shape != (batch_size, args.sample_q, 3):
                raise RuntimeError('Unexpected output shape.')
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Non-finite evaluation output.')
            diagnostic = candidate.last_primitive_diagnostics
            print(
                'scale={:g}, pred={}, baseline_diff={:.6f}, branches={}, '
                'opacity_min={:.6f}'.format(
                    scale_value,
                    tuple(prediction.shape),
                    (prediction - baseline_prediction).abs().mean().item(),
                    '/'.join('{:.6f}'.format(value) for value in
                             diagnostic['branch_response'].cpu().tolist()),
                    diagnostic['opacity_weights'].min().item(),
                )
            )

    candidate.train()
    height = args.inp_size * 4
    width = (args.inp_size + 2) * 4
    coord, cell = make_queries(
        height, width, args.sample_q, batch_size, device
    )
    scale = torch.full((batch_size,), 4.0, device=device)
    prediction = candidate(inp, coord, scale, cell)
    loss = prediction.square().mean()
    loss.backward()
    gradients = {
        'feature-head': gradient_mean(
            candidate.primitive_residual_head.weight, 'feature head'
        ),
        'offset': gradient_mean(candidate.offset_delta, 'offset'),
        'covariance': gradient_mean(
            candidate.covariance_scale_delta, 'covariance scale'
        ),
        'opacity': gradient_mean(candidate.opacity_logit, 'opacity'),
    }
    print('training loss: {:.6f}'.format(loss.item()))
    print('gradients:', '/'.join(
        '{}={:.3e}'.format(name, value) for name, value in gradients.items()
    ))

    chunk_spec = copy.deepcopy(candidate_spec)
    chunk_spec['args']['primitive_chunk_size'] = 1
    chunk_one = models.make(chunk_spec).to(device).eval()
    chunk_one.load_state_dict(candidate.state_dict())
    candidate.eval()
    with torch.no_grad():
        reference = chunk_one(inp[:1], coord[:1], scale[:1], cell[:1])
        chunked = candidate(inp[:1], coord[:1], scale[:1], cell[:1])
        repeated = candidate(inp[:1], coord[:1], scale[:1], cell[:1])
    chunk_error = (reference - chunked).abs().max().item()
    repeat_error = (chunked - repeated).abs().max().item()
    if chunk_error > 1e-4 or repeat_error > 1e-7:
        raise RuntimeError('Chunk/repeat consistency check failed.')
    print('chunk/repeat tolerances: 1.00e-04/1.00e-07')
    print('chunk/repeat errors: {:.2e}/{:.2e}'.format(
        chunk_error, repeat_error
    ))
    print('MULTI-PRIMITIVE SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
