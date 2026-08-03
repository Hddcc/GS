import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import models
import utils
from models.gaussian import GaussianSplatter, make_coord

try:
    from test import batched_predict
except ModuleNotFoundError as error:
    if error.name != 'tqdm':
        raise

    def batched_predict(model, inp, coord, scale, cell, bsize):
        with torch.no_grad():
            model.gen_feat(inp)
            predictions = []
            for start in range(0, coord.shape[1], bsize):
                end = min(start + bsize, coord.shape[1])
                predictions.append(model.query_rgb(
                    coord[:, start:end, :].contiguous(),
                    scale.contiguous(),
                    cell[:, start:end, :].contiguous(),
                ))
            return torch.cat(predictions, dim=1)


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
    if value is None or not torch.isfinite(value).all():
        raise RuntimeError('{} is missing or contains NaN/Inf.'.format(name))


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


def legacy_decode_queries(model, coef, freq, coord, cell):
    query_grid = coord.flip(-1).unsqueeze(1)
    q_coef = F.grid_sample(
        coef, query_grid, mode='nearest', align_corners=False
    )[:, :, 0, :].permute(0, 2, 1)
    q_freq = F.grid_sample(
        freq, query_grid, mode='nearest', align_corners=False
    )[:, :, 0, :].permute(0, 2, 1)
    q_coord = F.grid_sample(
        model.feat_coord, query_grid, mode='nearest', align_corners=False
    )[:, :, 0, :].permute(0, 2, 1)
    relative_coord = coord - q_coord
    relative_coord = relative_coord.clone()
    relative_coord[:, :, 0] *= model.feat.shape[-2]
    relative_coord[:, :, 1] *= model.feat.shape[-1]
    relative_cell = cell.clone()
    relative_cell[:, :, 0] *= model.feat.shape[-2]
    relative_cell[:, :, 1] *= model.feat.shape[-1]
    batch_size, query_count = coord.shape[:2]
    q_freq = torch.stack(torch.split(q_freq, 2, dim=-1), dim=-1)
    q_freq = torch.mul(q_freq, relative_coord.unsqueeze(-1)).sum(dim=-2)
    q_freq += model.phase(
        relative_cell.view(batch_size * query_count, -1)
    ).view(batch_size, query_count, -1)
    q_freq = torch.cat((
        torch.cos(np.pi * q_freq),
        torch.sin(np.pi * q_freq),
    ), dim=-1)
    decoder_input = torch.mul(q_coef, q_freq)
    return model.dec(
        decoder_input.contiguous().view(batch_size * query_count, -1)
    ).view(batch_size, query_count, -1)


def check_default_decoder(model, device):
    height = width = 8
    query_count = 37
    model.feat = torch.rand(1, 64, height, width, device=device)
    model.feat_coord = make_coord(
        (height, width), flatten=False
    ).to(device).permute(2, 0, 1).unsqueeze(0)
    coef = torch.rand(1, 256, height, width, device=device)
    freq = torch.rand(1, 256, height, width, device=device)
    coord = utils.make_coord((9, 9)).to(device)[:query_count].unsqueeze(0)
    cell = torch.ones_like(coord) * (2 / 9)
    with torch.no_grad():
        legacy = legacy_decode_queries(model, coef, freq, coord, cell)
        refactored = GaussianSplatter.decode_queries(
            model, coef, freq, coord, cell
        )
    error = (legacy - refactored).abs().max().item()
    if error != 0:
        raise RuntimeError('Default decoder refactor changed its output.')
    return error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--scales', type=float, nargs='+', default=[1.5, 4, 8])
    parser.add_argument('--inp-size', type=int, default=16)
    parser.add_argument('--sample-q', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    device = torch.device(args.device)
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
    encoder = model.encoder
    default_error = check_default_decoder(model, device)
    print('model:', config['model']['name'])
    print('encoder:', config['model']['args']['encoder_spec']['name'])
    print('device:', device)
    print('seed:', seed)
    print('parameters:', sum(parameter.numel() for parameter in model.parameters()))
    print('default decoder refactor error: {:.2e}'.format(default_error))
    print('input/query:', args.inp_size, args.sample_q)

    for scale_value in args.scales:
        model.zero_grad(set_to_none=True)
        inp = torch.rand(
            1, 3, args.inp_size, args.inp_size, device=device
        ) * 2 - 1
        coord, cell = make_queries(
            args.inp_size, scale_value, args.sample_q, device
        )
        scale = torch.tensor([scale_value], device=device)
        pred = model(inp, coord, scale, cell)
        weights = model.last_ensemble_weights
        assert_finite('prediction', pred)
        assert_finite('ensemble weights', weights)
        assert_finite('RGB residual', model.last_rgb_residual)

        weight_error = (weights.sum(dim=-1) - 1).abs().max().item()
        weight_std = weights.std(dim=-1).mean().item()
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=-1).mean().item()
        if weight_error > 1e-5 or weight_std == 0:
            raise RuntimeError('Gaussian metric ensemble weights are invalid.')

        loss = pred.abs().mean()
        loss.backward()
        geometry_grad = gradient_norm(
            list(encoder.geometry_shallow.parameters())
            + list(encoder.geometry_middle.parameters())
            + list(encoder.geometry_head.parameters()),
            'geometry branch',
        )
        covariance_grad = gradient_norm(
            [model.sigma_x, model.sigma_y, model.rho],
            'Gaussian covariance',
        )
        residual_grad = gradient_norm(
            list(model.frequency_encoder.parameters())
            + list(model.residual_head.parameters())
            + [model.residual_gate_logit],
            'frequency residual branch',
        )

        usage = weights.mean(dim=(0, 1)).tolist()
        print(
            'scale={:g}, pred={}, loss={:.6f}, weight_error={:.2e}, '
            'usage=[{:.4f}/{:.4f}/{:.4f}/{:.4f}], weight_std={:.6f}, '
            'entropy={:.6f}, rgb_residual={:.8f}, '
            'grads=geometry:{:.3e}/covariance:{:.3e}/residual:{:.3e}'
            .format(
                scale_value,
                tuple(pred.shape),
                loss.item(),
                weight_error,
                usage[0],
                usage[1],
                usage[2],
                usage[3],
                weight_std,
                entropy,
                model.last_rgb_residual.abs().mean().item(),
                geometry_grad,
                covariance_grad,
                residual_grad,
            )
        )

    model.eval()
    with torch.no_grad():
        inp = torch.rand(1, 3, args.inp_size, args.inp_size, device=device) * 2 - 1
        coord, cell = make_queries(args.inp_size, 4, args.sample_q, device)
        scale = torch.tensor([4.0], device=device)
        output = model(inp, coord, scale, cell)
        batched_output = batched_predict(
            model,
            inp,
            coord,
            scale,
            cell,
            max(1, args.sample_q // 3),
        )
    assert_finite('evaluation output', output)
    assert_finite('batched evaluation output', batched_output)
    batched_error = (output - batched_output).abs().max().item()
    if batched_error > 1e-6:
        raise RuntimeError('Batched prediction changed model output.')
    print('evaluation output:', tuple(output.shape))
    print(
        'batched evaluation output: {}, max_error={:.2e}'.format(
            tuple(batched_output.shape), batched_error
        )
    )
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
