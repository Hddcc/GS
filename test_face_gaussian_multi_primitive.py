import random

import numpy as np
import torch

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_spec(name, chunk_size=7):
    args = {
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
        'primitive_chunk_size': chunk_size,
        'gaussian_channels': 8,
    }
    if name == 'gaussian-splatter-memory-efficient-m4':
        args.update({
            'primitive_count': 4,
            'residual_limit': 0.1,
            'offset_residual_limit': 0.25,
            'covariance_log_limit': 0.25,
        })
    return {'name': name, 'args': args}


def make_queries(height, width, count, batch_size=1):
    coord = utils.make_coord((height, width))
    indices = torch.linspace(0, len(coord) - 1, steps=count).long()
    coord = coord[indices].unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


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


def finite_active_gradient(parameter, label):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError('{} has no finite gradient.'.format(label))
    value = parameter.grad.abs().mean().item()
    if value <= 0:
        raise RuntimeError('{} gradient is zero.'.format(label))
    return value


def main():
    seed_all(47)
    baseline = models.make(
        make_spec('gaussian-splatter-memory-efficient')
    )
    seed_all(47)
    candidate = models.make(
        make_spec('gaussian-splatter-memory-efficient-m4')
    )
    initialization_error, compared = shared_initialization_error(
        baseline, candidate
    )
    if initialization_error != 0:
        raise RuntimeError('Shared initialization changed.')
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    if candidate_parameters - baseline_parameters != 2100:
        raise RuntimeError('Unexpected parameter delta.')
    expected_offsets = torch.tensor([
        [-0.25, -0.25],
        [-0.25, 0.25],
        [0.25, -0.25],
        [0.25, 0.25],
    ])
    if not torch.equal(candidate.primitive_offsets(), expected_offsets):
        raise RuntimeError('Initial primitive offsets are wrong.')
    if not torch.equal(candidate.covariance_scales(), torch.ones(4, 2)):
        raise RuntimeError('Initial covariance scales are wrong.')
    if not torch.allclose(
            candidate.opacity_weights(), torch.full((4,), 0.25),
            rtol=0, atol=1e-7):
        raise RuntimeError('Initial opacity weights are wrong.')

    inp = torch.rand(1, 3, 6, 9) * 2 - 1
    coord, cell = make_queries(9, 14, 41)
    scale = torch.tensor([1.5])
    candidate.train()
    output = candidate(inp, coord, scale, cell)
    if output.shape != (1, 41, 3) or not torch.isfinite(output).all():
        raise RuntimeError('Candidate output is invalid.')
    output.square().mean().backward()
    gradients = {
        'feature-head': finite_active_gradient(
            candidate.primitive_residual_head.weight, 'feature head'
        ),
        'offset': finite_active_gradient(candidate.offset_delta, 'offset'),
        'covariance': finite_active_gradient(
            candidate.covariance_scale_delta, 'covariance scale'
        ),
        'opacity': finite_active_gradient(
            candidate.opacity_logit, 'opacity'
        ),
    }
    diagnostic = candidate.last_primitive_diagnostics
    if diagnostic is None:
        raise RuntimeError('Primitive diagnostics are missing.')
    responses = diagnostic['branch_response']
    if responses.shape != (4,) or responses.min().item() <= 0:
        raise RuntimeError('A primitive color branch is inactive.')
    diversity = (
        responses.std() / responses.mean().clamp_min(1e-12)
    ).item()
    if diversity <= 0.01:
        raise RuntimeError('Primitive color branches are not diverse.')

    seed_all(53)
    chunk_one = models.make(
        make_spec('gaussian-splatter-memory-efficient-m4', chunk_size=1)
    ).eval()
    chunk_seven = models.make(
        make_spec('gaussian-splatter-memory-efficient-m4', chunk_size=7)
    ).eval()
    chunk_seven.load_state_dict(chunk_one.state_dict())
    with torch.no_grad():
        output_one = chunk_one(inp, coord, scale, cell)
        output_seven = chunk_seven(inp, coord, scale, cell)
        repeat = chunk_seven(inp, coord, scale, cell)
    chunk_error = (output_one - output_seven).abs().max().item()
    repeat_error = (output_seven - repeat).abs().max().item()
    if chunk_error > 2e-6 or repeat_error != 0:
        raise RuntimeError('Chunk/repeat consistency failed.')

    print('parameters: baseline={}, candidate={}, delta={}'.format(
        baseline_parameters,
        candidate_parameters,
        candidate_parameters - baseline_parameters,
    ))
    print('shared parameters compared:', compared)
    print('shared initialization error: {:.2e}'.format(initialization_error))
    print('branch responses:', '/'.join(
        '{:.8f}'.format(value) for value in responses.tolist()
    ))
    print('branch diversity: {:.8f}'.format(diversity))
    print('gradients:', '/'.join(
        '{}={:.3e}'.format(name, value) for name, value in gradients.items()
    ))
    print('chunk/repeat errors: {:.2e}/{:.2e}'.format(
        chunk_error, repeat_error
    ))
    print('MULTI-PRIMITIVE MECHANISM TEST PASSED')


if __name__ == '__main__':
    main()
