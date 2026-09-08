import argparse
import copy
import os
import random
import statistics
import time

import numpy as np
import torch

import models
import utils


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def candidate_spec(checkpoint):
    spec = copy.deepcopy(checkpoint['model'])
    spec['name'] = 'gaussian-splatter-memory-efficient'
    spec['args']['primitive_chunk_size'] = 7
    spec['args']['gaussian_channels'] = 8
    return spec


def make_query(height, width, count, device):
    coord = utils.make_coord((height, width)).to(device)
    indices = torch.linspace(0, len(coord) - 1, steps=count).long()
    coord = coord[indices].unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[..., 0] *= 2 / height
    cell[..., 1] *= 2 / width
    return coord, cell


def build(spec, device):
    return models.make(copy.deepcopy(spec), load_sd=True).to(device).eval()


def measure(spec, inp, coord, scale, cell, warmup, repeats):
    device = inp.device
    torch.cuda.empty_cache()
    model = build(spec, device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    output = None
    with torch.no_grad():
        for index in range(warmup + repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(inp, coord, scale, cell)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if index >= warmup:
                timings.append(elapsed)
    peak = torch.cuda.max_memory_allocated(device)
    layer_elements = getattr(model, 'last_peak_layer_elements', None)
    baseline_elements = getattr(model, 'last_baseline_layer_elements', None)
    output = output.detach().cpu()
    del model
    torch.cuda.empty_cache()
    return output, peak, statistics.median(timings), layer_elements, baseline_elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--input-height', type=int, default=16)
    parser.add_argument('--input-width', type=int, default=24)
    parser.add_argument('--scale', type=float, default=8.0)
    parser.add_argument('--queries', type=int, default=512)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--minimum-memory-reduction', type=float, default=0.30)
    parser.add_argument('--maximum-output-error', type=float, default=2e-5)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    if min(args.input_height, args.input_width, args.queries, args.repeats) <= 0:
        raise ValueError('Input sizes, queries, and repeats must be positive.')
    if args.scale <= 0 or args.warmup < 0:
        raise ValueError('Scale must be positive and warmup non-negative.')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if checkpoint['model']['name'] != 'gaussian-splatter':
        raise RuntimeError('Checkpoint is not a GaussianSR baseline checkpoint.')
    if checkpoint.get('seed') != args.seed:
        raise RuntimeError('Checkpoint seed does not match --seed.')
    baseline_spec = checkpoint['model']
    efficient_spec = candidate_spec(checkpoint)
    device = torch.device('cuda')
    seed_all(args.seed)
    inp = torch.rand(
        1, 3, args.input_height, args.input_width, device=device
    ) * 2 - 1
    output_h = round(args.input_height * args.scale)
    output_w = round(args.input_width * args.scale)
    coord, cell = make_query(
        output_h, output_w, args.queries, device
    )
    scale = torch.tensor([args.scale], device=device)

    baseline = measure(
        baseline_spec, inp, coord, scale, cell, args.warmup, args.repeats
    )
    efficient = measure(
        efficient_spec, inp, coord, scale, cell, args.warmup, args.repeats
    )
    output_error = (baseline[0] - efficient[0]).abs().max().item()
    memory_reduction = 1 - efficient[1] / baseline[1]
    speed_ratio = efficient[2] / baseline[2]
    if not np.isfinite(output_error) or output_error > args.maximum_output_error:
        raise RuntimeError(
            'Output error {:.3e} exceeds {:.3e}.'.format(
                output_error, args.maximum_output_error
            )
        )
    if memory_reduction < args.minimum_memory_reduction:
        raise RuntimeError(
            'Memory reduction {:.2%} is below {:.2%}.'.format(
                memory_reduction, args.minimum_memory_reduction
            )
        )
    if efficient[3] is None or efficient[4] is None:
        raise RuntimeError('Candidate layer accounting is unavailable.')

    print('checkpoint:', args.checkpoint)
    print('seed:', args.seed)
    print('synthetic input: {}x{}, scale x{:g}, queries {}'.format(
        args.input_height, args.input_width, args.scale, args.queries
    ))
    print('maximum output error: {:.3e}'.format(output_error))
    print('baseline peak allocated: {:.2f} MiB'.format(
        baseline[1] / 2 ** 20
    ))
    print('candidate peak allocated: {:.2f} MiB'.format(
        efficient[1] / 2 ** 20
    ))
    print('peak memory reduction: {:.2%}'.format(memory_reduction))
    print('baseline median forward: {:.4f} s'.format(baseline[2]))
    print('candidate median forward: {:.4f} s'.format(efficient[2]))
    print('candidate/baseline time ratio: {:.3f}'.format(speed_ratio))
    print('largest layer elements: baseline={} chunk7={}'.format(
        efficient[4], efficient[3]
    ))
    print('MEMORY-EFFICIENT RASTER CUDA SCREEN PASSED')


if __name__ == '__main__':
    main()
