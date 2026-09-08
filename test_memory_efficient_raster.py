import argparse
import copy
import os
import random

import numpy as np
import torch

import models
import utils


OUTPUT_TOLERANCE = 3e-6
GRADIENT_TOLERANCE = 5e-6


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_spec(name, chunk_size=None):
    args = {
        'encoder_spec': {
            'name': 'edsr-baseline',
            'args': {
                'n_resblocks': 2,
                'n_feats': 16,
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
    }
    if chunk_size is not None:
        args.update({
            'primitive_chunk_size': chunk_size,
            'gaussian_channels': 8,
        })
    return {'name': name, 'args': args}


def make_queries(height, width, count, batch_size, device):
    coord = utils.make_coord((height, width)).to(device)
    indices = torch.linspace(0, len(coord) - 1, steps=count).long()
    coord = coord[indices].unsqueeze(0).expand(
        batch_size, -1, -1
    ).contiguous()
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


def maximum_error(left, right):
    return (left - right).abs().max().item()


def assert_close(name, left, right, tolerance):
    error = maximum_error(left, right)
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            '{} error {:.3e} exceeds {:.3e}.'.format(
                name, error, tolerance
            )
        )
    return error


def run_forward(model, inp, coord, scale, cell, seed):
    seed_all(seed)
    return model(inp, coord, scale, cell)


def parameter_gradients(model):
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            gradients[name] = None
        else:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Non-finite gradient: {}.'.format(name))
            gradients[name] = parameter.grad.detach().clone()
    return gradients


def compare_gradients(baseline, candidate):
    baseline_gradients = parameter_gradients(baseline)
    candidate_gradients = parameter_gradients(candidate)
    maximum = 0.0
    active = 0
    for name, baseline_gradient in baseline_gradients.items():
        candidate_gradient = candidate_gradients[name]
        if baseline_gradient is None or candidate_gradient is None:
            if baseline_gradient is not None or candidate_gradient is not None:
                raise RuntimeError('Gradient presence differs at {}.'.format(name))
            continue
        active += int(baseline_gradient.abs().sum().item() > 0)
        maximum = max(maximum, maximum_error(
            baseline_gradient, candidate_gradient
        ))
    if active == 0:
        raise RuntimeError('No active parameter gradient was observed.')
    if maximum > GRADIENT_TOLERANCE:
        raise RuntimeError(
            'Gradient error {:.3e} exceeds {:.3e}.'.format(
                maximum, GRADIENT_TOLERANCE
            )
        )
    return maximum, active


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')
    device = torch.device(args.device)

    seed_all(args.seed)
    baseline = models.make(make_spec('gaussian-splatter')).to(device)
    candidates = []
    for chunk_size in (1, 7, 49):
        candidate = models.make(make_spec(
            'gaussian-splatter-memory-efficient', chunk_size
        )).to(device)
        candidate.load_state_dict(copy.deepcopy(baseline.state_dict()))
        candidates.append(candidate)

    batch_size = 2
    inp = torch.rand(batch_size, 3, 6, 9, device=device) * 2 - 1
    output_h, output_w = 9, 14
    coord, cell = make_queries(
        output_h, output_w, 41, batch_size, device
    )
    scale = torch.full((batch_size,), 1.5, device=device)

    baseline.eval()
    for candidate in candidates:
        candidate.eval()
    with torch.no_grad():
        reference = run_forward(
            baseline, inp, coord, scale, cell, args.seed + 10
        )
        output_errors = []
        for candidate in candidates:
            output = run_forward(
                candidate, inp, coord, scale, cell, args.seed + 10
            )
            output_errors.append(assert_close(
                'eval output, chunk={}'.format(
                    candidate.primitive_chunk_size
                ),
                reference,
                output,
                OUTPUT_TOLERANCE,
            ))

    baseline.train()
    candidate = candidates[1].train()
    baseline.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    baseline_output = run_forward(
        baseline, inp, coord, scale, cell, args.seed + 20
    )
    candidate_output = run_forward(
        candidate, inp, coord, scale, cell, args.seed + 20
    )
    train_error = assert_close(
        'seeded train output', baseline_output, candidate_output,
        OUTPUT_TOLERANCE,
    )
    target = torch.linspace(
        -0.7, 0.8, steps=baseline_output.numel(), device=device
    ).view_as(baseline_output)
    (baseline_output - target).square().mean().backward()
    (candidate_output - target).square().mean().backward()
    gradient_error, active_gradients = compare_gradients(
        baseline, candidate
    )

    expected_baseline_elements = candidate.last_baseline_layer_elements
    expected_chunk_elements = candidate.last_peak_layer_elements
    if expected_chunk_elements * 7 != expected_baseline_elements:
        raise RuntimeError(
            'Chunked layer accounting is inconsistent: {} vs {}.'.format(
                expected_chunk_elements, expected_baseline_elements
            )
        )

    print('device:', device)
    print('non-square input/output: 6x9 -> 9x14')
    print('eval max errors for chunks 1/7/49:', ' '.join(
        '{:.3e}'.format(value) for value in output_errors
    ))
    print('train output max error: {:.3e}'.format(train_error))
    print('gradient max error: {:.3e}'.format(gradient_error))
    print('active parameter gradients:', active_gradients)
    print('largest layer elements: baseline={} chunk7={}'.format(
        expected_baseline_elements, expected_chunk_elements
    ))
    print('MEMORY-EFFICIENT RASTER EQUIVALENCE TEST PASSED')


if __name__ == '__main__':
    main()
