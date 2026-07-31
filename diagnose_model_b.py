import argparse
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import datasets
import models


torch.backends.cudnn.enabled = False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Diagnose the learned frequency/scale Gaussian modulation without '
            'changing the checkpoint.'
        )
    )
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--scales', type=float, nargs='+', default=[2, 4, 8])
    parser.add_argument('--reference-scale', type=float, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--num-images', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data_norm(model_path, config_path):
    if config_path is None:
        candidate = Path(model_path).resolve().parent / 'config.yaml'
        config_path = candidate if candidate.exists() else None
    if config_path is None:
        return {'inp': {'sub': [0.5], 'div': [0.5]}}
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config.get('data_norm', {
        'inp': {'sub': [0.5], 'div': [0.5]},
    })


def make_wrapped_dataset(base_dataset, scale):
    return datasets.make({
        'name': 'sr-implicit-downsampled',
        'args': {
            'inp_size': None,
            'scale_min': scale,
            'scale_max': scale,
            'augment': False,
            'sample_q': None,
            'batch_per_gpu': 1,
        },
    }, args={'dataset': base_dataset})


def pearson(x, y):
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum())
    if denominator.item() <= 1e-12:
        return float('nan')
    return float((x * y).sum().div(denominator).item())


def add_stats(row, prefix, value):
    value = value.float().reshape(-1)
    row[prefix + '_mean'] = float(value.mean().item())
    row[prefix + '_std'] = float(value.std(unbiased=False).item())
    row[prefix + '_abs_mean'] = float(value.abs().mean().item())
    row[prefix + '_abs_max'] = float(value.abs().max().item())


def bounded_deltas(model, raw_residuals):
    if hasattr(model, 'bounded_gaussian_deltas'):
        return model.bounded_gaussian_deltas(raw_residuals)
    return {
        'delta_sigma_x': (
            model.max_log_sigma_delta * torch.tanh(raw_residuals[:, 0])
        ),
        'delta_sigma_y': (
            model.max_log_sigma_delta * torch.tanh(raw_residuals[:, 1])
        ),
        'delta_rho': model.max_rho_delta * torch.tanh(raw_residuals[:, 2]),
        'delta_opacity': (
            model.max_opacity_logit_delta * torch.tanh(raw_residuals[:, 3])
        ),
    }


def directional_frequency(model, patch_features):
    if hasattr(model.local_modulator, 'directional_frequency'):
        return model.local_modulator.directional_frequency(patch_features)
    gray = patch_features.mean(dim=1, keepdim=True)
    gray = F.pad(gray, (0, 1, 0, 1), mode='replicate')
    directional = F.conv2d(
        gray,
        model.local_modulator.haar_filters.to(dtype=gray.dtype),
    ).abs()
    energy = torch.sqrt(
        directional.square().sum(dim=1, keepdim=True) + 1e-8
    )
    return directional, energy[:, 0]


def saturation_ratios(model, raw_residuals):
    if hasattr(model, 'residual_saturation_ratios'):
        return model.residual_saturation_ratios(raw_residuals)
    channel_by_name = {
        'delta_sigma_x': 0,
        'delta_sigma_y': 1,
        'delta_rho': 2,
        'delta_opacity': 3,
    }
    return {
        name: float(
            (torch.tanh(raw_residuals[:, channel]).abs() > 0.95)
            .float().mean().item()
        )
        for name, channel in channel_by_name.items()
    }


def gaussian_geometry(sigma_x, sigma_y, rho):
    covariance_x = sigma_x.square()
    covariance_y = sigma_y.square()
    covariance_xy = rho * sigma_x * sigma_y
    trace = covariance_x + covariance_y
    discriminant = torch.sqrt(
        (covariance_x - covariance_y).square()
        + 4 * covariance_xy.square()
        + 1e-12
    )
    eigenvalue_max = ((trace + discriminant) / 2).clamp_min(1e-8)
    eigenvalue_min = ((trace - discriminant) / 2).clamp_min(1e-8)
    effective_size = (eigenvalue_max * eigenvalue_min).pow(0.25)
    log_anisotropy = 0.5 * torch.log(eigenvalue_max / eigenvalue_min)
    return effective_size, log_anisotropy


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def related_path(output_path, suffix):
    output_path = Path(output_path)
    return output_path.with_name(output_path.stem + suffix + output_path.suffix)


def summarize(per_image_rows):
    groups = defaultdict(list)
    for row in per_image_rows:
        groups[(row['dataset'], row['seed'], row['scale'])].append(row)

    identifier_fields = {'dataset', 'image', 'seed', 'scale'}
    summaries = []
    for (dataset_name, seed, scale), rows in sorted(groups.items()):
        summary = {
            'dataset': dataset_name,
            'seed': seed,
            'scale': scale,
            'images': len(rows),
        }
        for key in rows[0]:
            if key in identifier_fields:
                continue
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            if np.all(np.isnan(values)):
                summary[key] = float('nan')
            else:
                summary[key] = float(np.nanmean(values))
        summaries.append(summary)
    return summaries


def summarize_scale_only(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row['dataset'], row['seed'], row['input_scale'], row['target_scale']
        )].append(row)
    summaries = []
    identifiers = {'dataset', 'image', 'seed', 'input_scale', 'target_scale'}
    for key, group_rows in sorted(groups.items()):
        dataset_name, seed, input_scale, target_scale = key
        summary = {
            'dataset': dataset_name,
            'seed': seed,
            'input_scale': input_scale,
            'target_scale': target_scale,
            'images': len(group_rows),
        }
        for field in group_rows[0]:
            if field in identifiers:
                continue
            summary[field] = float(np.mean([
                row[field] for row in group_rows
            ]))
        summaries.append(summary)
    return summaries


def seed_variation(per_image_rows, assignments, seeds):
    if len(seeds) < 2:
        return []
    grouped_rows = defaultdict(list)
    for row in per_image_rows:
        grouped_rows[(row['dataset'], row['image'], row['scale'])].append(row)

    variation_rows = []
    tracked = [
        'delta_sigma_x_abs_mean',
        'delta_sigma_y_abs_mean',
        'delta_rho_abs_mean',
        'delta_opacity_abs_mean',
        'sigma_x_final_mean',
        'sigma_y_final_mean',
        'rho_final_mean',
        'opacity_final_mean',
    ]
    for key, rows in sorted(grouped_rows.items()):
        dataset_name, image_name, scale = key
        row = {
            'dataset': dataset_name,
            'image': image_name,
            'scale': scale,
            'seeds': len(rows),
        }
        for field in tracked:
            values = np.asarray([item[field] for item in rows], dtype=np.float64)
            row[field + '_range'] = float(values.max() - values.min())

        seed_maps = assignments[key]
        reference = seed_maps[seeds[0]]
        disagreement = []
        for seed in seeds[1:]:
            current = seed_maps[seed]
            disagreement.append(float(np.mean(reference != current)))
        row['gaussian_class_disagreement'] = float(np.mean(disagreement))
        variation_rows.append(row)
    return variation_rows


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for GaussianSR diagnostics.')
    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError(
            'Dataset directory not found: {}'.format(args.dataset_root)
        )
    if not os.path.isfile(args.model):
        raise FileNotFoundError('Checkpoint not found: {}'.format(args.model))
    if args.num_images < 1:
        raise ValueError('--num-images must be at least 1.')
    if args.reference_scale not in args.scales:
        raise ValueError('--reference-scale must be included in --scales.')

    checkpoint = torch.load(args.model, map_location='cpu')
    model = models.make(checkpoint['model'], load_sd=True).cuda().eval()
    if not hasattr(model, 'local_modulator'):
        raise TypeError(
            'The checkpoint is not a local frequency-scale Gaussian model.'
        )

    data_norm = load_data_norm(args.model, args.config)
    inp_cfg = data_norm['inp']
    inp_sub = torch.tensor(
        inp_cfg['sub'], dtype=torch.float32, device='cuda'
    ).view(1, -1, 1, 1)
    inp_div = torch.tensor(
        inp_cfg['div'], dtype=torch.float32, device='cuda'
    ).view(1, -1, 1, 1)

    base_dataset = datasets.make({
        'name': 'image-folder',
        'args': {'root_path': args.dataset_root, 'cache': 'none'},
    })
    filenames = [Path(path).name for path in base_dataset.files]
    image_count = min(args.num_images, len(base_dataset))
    indices = np.linspace(
        0, len(base_dataset) - 1, image_count, dtype=np.int64
    ).tolist()
    wrapped = {
        scale: make_wrapped_dataset(base_dataset, scale)
        for scale in args.scales
    }

    per_image_rows = []
    scale_only_rows = []
    assignments = defaultdict(dict)

    for seed in args.seeds:
        set_seed(seed)
        for scale in args.scales:
            for image_index in indices:
                sample = wrapped[scale][image_index]
                image_name = filenames[image_index]
                inp = sample['inp'].unsqueeze(0).cuda()
                inp = (inp - inp_sub) / inp_div
                coord = sample['coord'][:1].unsqueeze(0).cuda()
                cell = sample['cell'][:1].unsqueeze(0).cuda()
                scale_tensor = torch.tensor(
                    [float(scale)], dtype=torch.float32, device='cuda'
                )

                captured = {}

                def capture_modulator(module, inputs, output):
                    captured['patch_features'] = inputs[0].detach()
                    captured['raw_residuals'] = output.detach()

                handle = model.local_modulator.register_forward_hook(
                    capture_modulator
                )
                with torch.no_grad():
                    model.gen_feat(inp)
                    assignment = model.logits.argmax(dim=1).detach().cpu().numpy()
                    model.query_rgb(coord, scale_tensor, cell)
                handle.remove()

                if not captured:
                    raise RuntimeError('The local modulator hook was not called.')

                patch_features = captured['patch_features']
                raw_residuals = captured['raw_residuals']
                deltas = bounded_deltas(model, raw_residuals)
                saturation = saturation_ratios(model, raw_residuals)
                directional, energy = directional_frequency(model, patch_features)

                final_parameters = [
                    value.detach() for value in model.last_gaussian_parameters
                ]
                sigma_x, sigma_y, rho, opacity = final_parameters
                delta_sigma_x = deltas['delta_sigma_x'].reshape_as(sigma_x)
                delta_sigma_y = deltas['delta_sigma_y'].reshape_as(sigma_y)
                delta_rho = deltas['delta_rho'].reshape_as(rho)
                delta_opacity = deltas['delta_opacity'].reshape_as(opacity)
                energy_flat = energy.reshape_as(sigma_x)

                sigma_x_base = sigma_x / torch.exp(delta_sigma_x)
                sigma_y_base = sigma_y / torch.exp(delta_sigma_y)
                rho_base_unclamped = rho - delta_rho
                opacity_base = torch.sigmoid(
                    torch.logit(opacity.clamp(1e-6, 1 - 1e-6))
                    - delta_opacity
                )
                effective_size, log_anisotropy = gaussian_geometry(
                    sigma_x, sigma_y, rho
                )

                row = {
                    'dataset': args.dataset_name,
                    'image': image_name,
                    'seed': seed,
                    'scale': scale,
                    'patches': int(patch_features.shape[0]),
                    'points': int(sigma_x.numel()),
                    'selected_gaussian_classes': int(np.unique(assignment).size),
                }
                add_stats(row, 'frequency_energy', energy_flat)
                add_stats(row, 'frequency_horizontal', directional[:, 0])
                add_stats(row, 'frequency_vertical', directional[:, 1])
                add_stats(row, 'frequency_diagonal', directional[:, 2])
                for name, delta in deltas.items():
                    add_stats(row, name, delta)
                    row[name + '_saturation_ratio'] = saturation[name]

                add_stats(row, 'sigma_x_base', sigma_x_base)
                add_stats(row, 'sigma_y_base', sigma_y_base)
                add_stats(row, 'rho_base_unclamped', rho_base_unclamped)
                add_stats(row, 'opacity_base', opacity_base)
                add_stats(row, 'sigma_x_final', sigma_x)
                add_stats(row, 'sigma_y_final', sigma_y)
                add_stats(row, 'rho_final', rho)
                add_stats(row, 'opacity_final', opacity)
                add_stats(row, 'effective_size', effective_size)
                add_stats(row, 'log_anisotropy', log_anisotropy)
                row['sigma_x_relative_change_abs_mean'] = float(
                    torch.expm1(delta_sigma_x).abs().mean().item()
                )
                row['sigma_y_relative_change_abs_mean'] = float(
                    torch.expm1(delta_sigma_y).abs().mean().item()
                )
                row['rho_clamp_ratio'] = float(
                    (rho.abs() >= 0.9499).float().mean().item()
                )
                row['corr_energy_effective_size'] = pearson(
                    energy_flat, effective_size
                )
                row['corr_energy_log_anisotropy'] = pearson(
                    energy_flat, log_anisotropy
                )
                row['corr_energy_abs_rho'] = pearson(
                    energy_flat, rho.abs()
                )
                per_image_rows.append(row)
                assignments[(
                    args.dataset_name, image_name, scale
                )][seed] = assignment

                if scale == args.reference_scale:
                    with torch.no_grad():
                        reference_raw = model.local_modulator(
                            patch_features,
                            torch.full(
                                (patch_features.shape[0],),
                                float(args.reference_scale),
                                dtype=patch_features.dtype,
                                device=patch_features.device,
                            ),
                        )
                        reference_deltas = bounded_deltas(model, reference_raw)
                        for target_scale in args.scales:
                            target_raw = model.local_modulator(
                                patch_features,
                                torch.full(
                                    (patch_features.shape[0],),
                                    float(target_scale),
                                    dtype=patch_features.dtype,
                                    device=patch_features.device,
                                ),
                            )
                            target_deltas = bounded_deltas(model, target_raw)
                            scale_row = {
                                'dataset': args.dataset_name,
                                'image': image_name,
                                'seed': seed,
                                'input_scale': args.reference_scale,
                                'target_scale': target_scale,
                            }
                            for name in reference_deltas:
                                scale_row[name + '_l1_from_reference'] = float(
                                    (
                                        target_deltas[name]
                                        - reference_deltas[name]
                                    ).abs().mean().item()
                                )
                            scale_only_rows.append(scale_row)

                print(
                    '{} seed={} x{:g} {}: |d_sigma|=({:.6f},{:.6f}), '
                    '|d_rho|={:.6f}, corr(E,size)={:.4f}'.format(
                        args.dataset_name,
                        seed,
                        scale,
                        image_name,
                        row['delta_sigma_x_abs_mean'],
                        row['delta_sigma_y_abs_mean'],
                        row['delta_rho_abs_mean'],
                        row['corr_energy_effective_size'],
                    )
                )

    summary_rows = summarize(per_image_rows)
    scale_only_summary = summarize_scale_only(scale_only_rows)
    variation_rows = seed_variation(
        per_image_rows, assignments, args.seeds
    )

    write_csv(args.output, summary_rows)
    write_csv(related_path(args.output, '_per_image'), per_image_rows)
    write_csv(related_path(args.output, '_scale_only'), scale_only_summary)
    write_csv(related_path(args.output, '_scale_only_per_image'), scale_only_rows)
    write_csv(related_path(args.output, '_seed_variation'), variation_rows)

    print('\nSummary:')
    for row in summary_rows:
        print(
            'seed={} x{:g}: |d_sigma|=({:.6f},{:.6f}), |d_rho|={:.6f}, '
            '|d_opacity|={:.6f}, sat=({:.4f},{:.4f},{:.4f},{:.4f}), '
            'corr(E,size)={:.4f}, corr(E,aniso)={:.4f}'.format(
                row['seed'],
                row['scale'],
                row['delta_sigma_x_abs_mean'],
                row['delta_sigma_y_abs_mean'],
                row['delta_rho_abs_mean'],
                row['delta_opacity_abs_mean'],
                row['delta_sigma_x_saturation_ratio'],
                row['delta_sigma_y_saturation_ratio'],
                row['delta_rho_saturation_ratio'],
                row['delta_opacity_saturation_ratio'],
                row['corr_energy_effective_size'],
                row['corr_energy_log_anisotropy'],
            )
        )

    print('\nOutput files:')
    for path in [
            Path(args.output),
            related_path(args.output, '_per_image'),
            related_path(args.output, '_scale_only'),
            related_path(args.output, '_scale_only_per_image'),
            related_path(args.output, '_seed_variation')]:
        if Path(path).exists():
            print(path)


if __name__ == '__main__':
    main()
