import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

import datasets
import models
from datasets.wrappers import resize_fn
from utils import make_coord


SCALES = (2., 2.5, 3., 3.5, 4.)


def predict(model, inp, coord, scale, cell, chunk):
    model.gen_feat(inp)
    return torch.cat([model.query_rgb(
        coord[:, i:i+chunk].contiguous(), scale, cell[:, i:i+chunk].contiguous(),
    ) for i in range(0, coord.shape[1], chunk)], dim=1)


def summarize_deltas(values):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(1)
    means = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()),
            'image_bootstrap_ci95': np.quantile(means, [.025, .975]).tolist(),
            'improved_image_fraction': float((values > 0).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--chunk', type=int, default=10000)
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    if args.samples <= 0 or args.chunk <= 0:
        raise ValueError('samples and chunk must be positive.')
    with Path('configs/generated/scale_gated_frequency/control.yaml').open() as file:
        config = yaml.safe_load(file)
    original = config['baseline_adaptation']['checkpoint']
    paths = {'original': original}
    original_hash = hashlib.sha256(Path(original).read_bytes()).hexdigest()
    for variant in ('control', 'residual', 'gated'):
        paths[variant] = 'save/face_scale_gated_frequency_' + variant + '_seed1/epoch-last.pth'
    loaded = {}
    for variant, path in paths.items():
        checkpoint = torch.load(path, map_location='cpu')
        if checkpoint.get('seed') != 1:
            raise ValueError('Unexpected checkpoint seed: ' + path)
        if variant != 'original' and checkpoint['epoch'] != config['epoch_max']:
            raise ValueError('A fine-tuning run is incomplete: ' + path)
        if variant != 'original':
            arm_config_path = Path('configs/generated/scale_gated_frequency') / (variant + '.yaml')
            with arm_config_path.open() as file:
                arm_config = yaml.safe_load(file)
            if (checkpoint.get('baseline_source', {}).get('sha256') != original_hash
                    or checkpoint['model']['name'] != arm_config['model']['name']
                    or checkpoint['model']['args'] != arm_config['model']['args']):
                raise ValueError('Checkpoint source or model does not match its arm: ' + path)
        loaded[variant] = models.make(checkpoint['model'], load_sd=True).to(args.device).eval()
    dataset = datasets.make(config['val_dataset']['dataset'])
    if len(dataset) < args.samples:
        raise ValueError('Not enough validation images.')
    rows = []
    with torch.no_grad():
        for index in range(args.samples):
            image = dataset[index]
            height, width = (value // 8 * 8 for value in image.shape[-2:])
            target_image = image[:, :height, :width].contiguous()
            target = target_image.reshape(3, -1).T.unsqueeze(0).to(args.device)
            coord = make_coord((height, width)).to(args.device).unsqueeze(0)
            cell = torch.ones_like(coord) * coord.new_tensor([2 / height, 2 / width])
            for value in SCALES:
                lr_size = (round(height / value), round(width / value))
                inp = resize_fn(target_image, lr_size).unsqueeze(0).to(args.device) * 2 - 1
                scale = coord.new_tensor([value])
                row = {'index': index, 'scale': value, 'lr_size': list(lr_size),
                       'hr_size': [height, width], 'psnr': {}, 'mechanism': {}}
                for variant, model in loaded.items():
                    prediction = (predict(model, inp, coord, scale, cell, args.chunk) / 2 + .5).clamp(0, 1)
                    if not torch.isfinite(prediction).all():
                        raise RuntimeError('Nonfinite predictions: ' + variant)
                    mse = (prediction - target).square().mean().item()
                    row['psnr'][variant] = -10 * math.log10(max(mse, 1e-12))
                    if variant in ('residual', 'gated'):
                        row['mechanism'][variant] = {
                            'last_chunk_gate_mean': model.last_scale_gate.mean().item(),
                            'last_chunk_gate_std': model.last_scale_gate.std(unbiased=False).item(),
                            'last_chunk_residual_mean_abs': model.last_rgb_residual.abs().mean().item(),
                        }
                rows.append(row)
            if (index + 1) % 10 == 0:
                print('EVALUATION: {}/{} images'.format(index + 1, args.samples), flush=True)
    report = {'status': 'exploratory validation only', 'metric': 'float RGB PSNR, no border shave',
              'ci_note': 'Image bootstrap uncertainty, not training-seed uncertainty.',
              'checkpoint_sha256': {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                    for name, path in paths.items()},
              'scales': {}, 'per_image': rows}
    print('scale | original | control | residual | gated | gated-control | gated-residual')
    for value in SCALES:
        selected = [row for row in rows if row['scale'] == value]
        means = {variant: float(np.mean([row['psnr'][variant] for row in selected]))
                 for variant in loaded}
        delta_control = summarize_deltas([row['psnr']['gated'] - row['psnr']['control'] for row in selected])
        delta_residual = summarize_deltas([row['psnr']['gated'] - row['psnr']['residual'] for row in selected])
        report['scales'][str(value)] = {'psnr': means, 'gated_minus_control': delta_control,
                                       'gated_minus_residual': delta_residual}
        print('{:g} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:+.5f} | {:+.5f}'.format(
            value, *means.values(), delta_control['mean'], delta_residual['mean']), flush=True)
    report['range_summary'] = {}
    for comparison in ('control', 'residual'):
        image_means = [np.mean([
            row['psnr']['gated'] - row['psnr'][comparison] for row in rows
            if row['index'] == index
        ]) for index in range(args.samples)]
        summary = summarize_deltas(image_means)
        report['range_summary']['gated_minus_' + comparison] = summary
        print('x2-x4 gated-{} mean delta: {:+.5f}; image CI95: {}'.format(
            comparison, summary['mean'], summary['image_bootstrap_ci95']), flush=True)
    output = Path('results/scale_gated_frequency_evaluation.json')
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SCALE-GATED FREQUENCY EXPLORATORY EVALUATION COMPLETED')
    print('JSON=' + str(output))


if __name__ == '__main__':
    main()
