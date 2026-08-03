import argparse
import math
import os
import random

import numpy as np
import torch
import yaml

import models
import utils

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
    print(
        'initial gates: appearance={:.6f}, residual={:.6f}'.format(
            torch.sigmoid(encoder.appearance_gate_logit).item(),
            torch.sigmoid(model.residual_gate_logit).item(),
        )
    )

    scale_logits = []
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
        assert_finite('prediction', pred)
        assert_finite('geometry probabilities', encoder.last_geometry_probabilities)
        assert_finite('depth weights', encoder.last_depth_weights)
        assert_finite('depth entropy', encoder.last_depth_entropy)
        assert_finite('scale logits', encoder.last_scale_logits)
        assert_finite('appearance residual', encoder.last_appearance_residual)
        assert_finite('RGB residual', model.last_rgb_residual)

        probability_error = (
            encoder.last_geometry_probabilities.sum(dim=1) - 1
        ).abs().max().item()
        routing_error = (
            encoder.last_depth_weights.sum(dim=1) - 1
        ).abs().max().item()
        if max(probability_error, routing_error) > 1e-5:
            raise RuntimeError('A probability distribution does not sum to one.')
        routing_spatial_std = encoder.last_depth_weights.flatten(2).std(
            dim=-1
        ).mean().item()
        if routing_spatial_std == 0:
            raise RuntimeError('Depth routing is spatially constant.')
        scale_logits.append(encoder.last_scale_logits.cpu())

        loss = pred.abs().mean()
        loss.backward()
        geometry_grad = gradient_norm(
            list(encoder.geometry_shallow.parameters())
            + list(encoder.geometry_middle.parameters())
            + list(encoder.geometry_head.parameters()),
            'geometry branch',
        )
        content_grad = gradient_norm(
            encoder.content_router.parameters(),
            'content router',
        )
        scale_grad = gradient_norm(
            encoder.scale_router.parameters(),
            'scale router',
        )
        gate_grad = gradient_norm(
            [encoder.appearance_gate_logit],
            'appearance gate',
        )
        residual_grad = gradient_norm(
            list(model.frequency_encoder.parameters())
            + list(model.residual_head.parameters())
            + [model.residual_gate_logit],
            'frequency residual branch',
        )

        usage = encoder.last_depth_weights.mean(dim=(0, 2, 3)).tolist()
        print(
            'scale={:g}, pred={}, loss={:.6f}, prob_error={:.2e}, '
            'route_error={:.2e}, usage=[{:.4f}/{:.4f}/{:.4f}], '
            'route_std={:.8f}, route_entropy={:.6f}, '
            'appearance={:.8f}, rgb_residual={:.8f}, '
            'grads=geometry:{:.3e}/content:{:.3e}/scale:{:.3e}/'
            'gate:{:.3e}/residual:{:.3e}'
            .format(
                scale_value,
                tuple(pred.shape),
                loss.item(),
                probability_error,
                routing_error,
                usage[0],
                usage[1],
                usage[2],
                routing_spatial_std,
                encoder.last_depth_entropy.mean().item(),
                encoder.last_appearance_residual.abs().mean().item(),
                model.last_rgb_residual.abs().mean().item(),
                geometry_grad,
                content_grad,
                scale_grad,
                gate_grad,
                residual_grad,
            )
        )

    scale_logit_gap = (scale_logits[0] - scale_logits[-1]).abs().max().item()
    if scale_logit_gap == 0:
        raise RuntimeError('Scale conditioning does not change routing logits.')
    print('endpoint scale-logit gap: {:.8f}'.format(scale_logit_gap))

    model.eval()
    with torch.no_grad():
        inp = torch.rand(1, 3, args.inp_size, args.inp_size, device=device) * 2 - 1
        coord, cell = make_queries(args.inp_size, 4, args.sample_q, device)
        scale = torch.tensor([4.0], device=device)
        output = model(
            inp,
            coord,
            scale,
            cell,
        )
        batched_output = batched_predict(
            model,
            inp,
            coord,
            scale,
            cell,
            max(1, args.sample_q // 3),
        )
    if not torch.is_tensor(output):
        raise RuntimeError('Evaluation forward must return a tensor.')
    assert_finite('evaluation output', output)
    assert_finite('batched evaluation output', batched_output)
    batched_error = (output - batched_output).abs().max().item()
    if batched_error > 1e-6:
        raise RuntimeError('Batched prediction changed model output.')
    print('evaluation output:', tuple(output.shape))
    print(
        'batched evaluation output: {}, max_error={:.2e}'.format(
            tuple(batched_output.shape),
            batched_error,
        )
    )
    print('SMOKE TEST PASSED')


if __name__ == '__main__':
    main()
