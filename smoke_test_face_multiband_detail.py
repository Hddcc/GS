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
    print('model:', config['model']['name'])
    print('encoder:', config['model']['args']['encoder_spec']['name'])
    print('device:', device)
    print('seed:', seed)
    print('parameters:', sum(parameter.numel() for parameter in model.parameters()))
    print('input/query:', args.inp_size, args.sample_q)
    kernel_error = max(
        abs(encoder.gaussian_kernel_3.sum().item() - 1),
        abs(encoder.gaussian_kernel_5.sum().item() - 1),
    )
    if kernel_error > 1e-6:
        raise RuntimeError('Gaussian detail kernels are not normalized.')
    print('kernel normalization error: {:.2e}'.format(kernel_error))
    print(
        'initial gates: appearance={:.6f}, residual={:.6f}'.format(
            torch.sigmoid(encoder.appearance_gate_logit).item(),
            torch.sigmoid(model.residual_gate_logit).item(),
        )
    )

    for scale_value in args.scales:
        model.zero_grad(set_to_none=True)
        inp = torch.rand(
            1, 3, args.inp_size, args.inp_size, device=device
        ) * 2 - 1
        coord, cell = make_queries(
            args.inp_size, scale_value, args.sample_q, device
        )
        pred = model(
            inp,
            coord,
            torch.tensor([scale_value], device=device),
            cell,
        )
        assert_finite('prediction', pred)
        assert_finite('geometry probabilities', encoder.last_geometry_probabilities)
        assert_finite('high band', encoder.last_high_band)
        assert_finite('middle band', encoder.last_middle_band)
        assert_finite('detail feature', encoder.last_detail_feature)
        assert_finite('appearance residual', encoder.last_appearance_residual)
        assert_finite('RGB residual', model.last_rgb_residual)

        probability_error = (
            encoder.last_geometry_probabilities.sum(dim=1) - 1
        ).abs().max().item()
        if probability_error > 1e-5:
            raise RuntimeError('Geometry probabilities do not sum to one.')
        high_energy = encoder.last_high_band.abs().mean().item()
        middle_energy = encoder.last_middle_band.abs().mean().item()
        detail_magnitude = encoder.last_detail_feature.abs().mean().item()
        if min(high_energy, middle_energy, detail_magnitude) == 0:
            raise RuntimeError('Multi-band detail representation is zero.')

        loss = pred.abs().mean()
        loss.backward()
        geometry_grad = gradient_norm(
            list(encoder.geometry_shallow.parameters())
            + list(encoder.geometry_middle.parameters())
            + list(encoder.geometry_head.parameters()),
            'geometry branch',
        )
        detail_grad = gradient_norm(
            encoder.detail_encoder.parameters(),
            'detail branch',
        )
        appearance_grad = gradient_norm(
            encoder.appearance_fusion.parameters(),
            'appearance branch',
        )
        appearance_gate_grad = gradient_norm(
            [encoder.appearance_gate_logit],
            'appearance gate',
        )
        residual_grad = gradient_norm(
            list(model.frequency_encoder.parameters())
            + list(model.residual_head.parameters())
            + [model.residual_gate_logit],
            'frequency residual branch',
        )

        print(
            'scale={:g}, pred={}, loss={:.6f}, prob_error={:.2e}, '
            'bands=high:{:.8f}/middle:{:.8f}, detail={:.8f}, '
            'appearance={:.8f}, rgb_residual={:.8f}, '
            'grads=geometry:{:.3e}/detail:{:.3e}/appearance:{:.3e}/'
            'gate:{:.3e}/residual:{:.3e}'
            .format(
                scale_value,
                tuple(pred.shape),
                loss.item(),
                probability_error,
                high_energy,
                middle_energy,
                detail_magnitude,
                encoder.last_appearance_residual.abs().mean().item(),
                model.last_rgb_residual.abs().mean().item(),
                geometry_grad,
                detail_grad,
                appearance_grad,
                appearance_gate_grad,
                residual_grad,
            )
        )

    model.eval()
    with torch.no_grad():
        inp = torch.rand(1, 3, args.inp_size, args.inp_size, device=device) * 2 - 1
        coord, cell = make_queries(args.inp_size, 4, args.sample_q, device)
        output = model(
            inp,
            coord,
            torch.tensor([4.0], device=device),
            cell,
        )
    if not torch.is_tensor(output):
        raise RuntimeError('Evaluation forward must return a tensor.')
    assert_finite('evaluation output', output)
    print('evaluation output:', tuple(output.shape))
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
