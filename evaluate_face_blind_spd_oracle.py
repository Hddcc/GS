import argparse
import copy
import json
import math
import os
import statistics

import torch
import yaml

import datasets
import models
from blind_spd_oracle import (
    apply_anisotropic_blur,
    calibrate_renderer_units,
    covariance_from_parameters,
    expand_manifest,
    gaussian_kernel,
    mapped_internal_covariance,
    oracle_variants,
    sha256_file,
)
from datasets.wrappers import resize_fn
from utils import make_coord


def load_model(path, expected_seed, device):
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('seed') != expected_seed:
        raise RuntimeError('{} has the wrong seed.'.format(path))
    model = models.make(checkpoint['model'], load_sd=True).to(device).eval()
    if not hasattr(model, 'set_oracle_covariance_adjustment'):
        raise RuntimeError('Baseline model lacks the oracle covariance hook.')
    return model


def make_dataset(spec, root_override=None):
    spec = copy.deepcopy(spec)
    spec.setdefault('args', {})['cache'] = 'none'
    if root_override is not None:
        spec['args']['root_path'] = root_override
    return datasets.make(spec)


def make_query(height, width, device):
    coord = make_coord((height, width)).to(device).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / height
    cell[:, :, 1] *= 2 / width
    return coord, cell


def query_from_features(model, coord, scale, cell, eval_bsize):
    predictions = []
    for start in range(0, coord.shape[1], eval_bsize):
        stop = min(start + eval_bsize, coord.shape[1])
        predictions.append(model.query_rgb(
            coord[:, start:stop].contiguous(),
            scale,
            cell[:, start:stop].contiguous(),
        ))
    return torch.cat(predictions, dim=1)


def psnr(prediction, target):
    mse = (prediction - target).square().mean().item()
    value = -10 * math.log10(max(mse, 1e-12))
    if not math.isfinite(value):
        raise RuntimeError('PSNR is not finite.')
    return value


def grouped_mean(records, key, value):
    values = [record['delta'] for record in records if record[key] == value]
    if not values:
        raise RuntimeError('Missing metric bucket {}={}.'.format(key, value))
    return statistics.mean(values)


def summarize(records, gated_scales, maximum_clamp_fraction):
    if not records:
        raise RuntimeError('Cannot summarize an empty evaluation.')
    summary = {
        'count': len(records),
        'overall_delta': statistics.mean(record['delta'] for record in records),
        'sample_std': statistics.stdev(record['delta'] for record in records)
            if len(records) > 1 else 0.0,
        'scale_delta': {},
        'bucket_delta': {},
        'maximum_clamp_fraction': max(
            record['clamp_fraction'] for record in records
        ),
        'minimum_covariance_eigenvalue': min(
            record['minimum_eigenvalue'] for record in records
        ),
        'maximum_covariance_eigenvalue': max(
            record['maximum_eigenvalue'] for record in records
        ),
    }
    for scale in sorted(set(record['scale'] for record in records)):
        summary['scale_delta']['{:g}'.format(scale)] = grouped_mean(
            records, 'scale', scale
        )
    for bucket in sorted(set(record['bucket'] for record in records)):
        summary['bucket_delta'][bucket] = grouped_mean(
            records, 'bucket', bucket
        )
    required_scale_values = [
        summary['scale_delta']['{:g}'.format(scale)]
        for scale in gated_scales
    ]
    summary['passes_gate'] = bool(
        summary['overall_delta'] > 0
        and summary['bucket_delta'].get('mild', -math.inf) > 0
        and summary['bucket_delta'].get('medium', -math.inf) > 0
        and all(value >= 0 for value in required_scale_values)
        and summary['maximum_clamp_fraction'] <= maximum_clamp_fraction
        and summary['minimum_covariance_eigenvalue'] > 0
        and math.isfinite(summary['maximum_covariance_eigenvalue'])
    )
    return summary


def print_summary(label, summary):
    print('{} count: {}'.format(label, summary['count']))
    print('{} overall paired delta: {:+.8f}'.format(
        label, summary['overall_delta']))
    print('{} paired sample std: {:.8f}'.format(label, summary['sample_std']))
    for scale, value in summary['scale_delta'].items():
        print('{} x{} delta: {:+.8f}'.format(label, scale, value))
    for bucket, value in summary['bucket_delta'].items():
        print('{} {} delta: {:+.8f}'.format(label, bucket, value))
    print('{} maximum clamp fraction: {:.8f}'.format(
        label, summary['maximum_clamp_fraction']))
    print('{} covariance eigenvalue range: {:.8f} .. {:.8f}'.format(
        label,
        summary['minimum_covariance_eigenvalue'],
        summary['maximum_covariance_eigenvalue'],
    ))
    print('{} gate: {}'.format(
        label, 'PASS' if summary['passes_gate'] else 'FAIL'))


def evaluate_split(model, dataset, entries, scales, variants, config,
                   protocol, device, eval_bsize, label):
    inp_norm = config['data_norm']['inp']
    gt_norm = config['data_norm']['gt']
    inp_sub = torch.tensor(inp_norm['sub'], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor(inp_norm['div'], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor(gt_norm['sub'], device=device).view(1, 1, -1)
    gt_div = torch.tensor(gt_norm['div'], device=device).view(1, 1, -1)
    renderer_units = calibrate_renderer_units(
        protocol['renderer_kernel_size'],
        protocol['renderer_reference_variance'],
        protocol['renderer_calibration_epsilon'],
    )
    records = {variant['name']: [] for variant in variants}
    realized_manifest = []
    with torch.no_grad():
        for entry_index, entry in enumerate(entries):
            index = entry['dataset_index']
            if index >= len(dataset):
                raise RuntimeError(
                    '{} index {} exceeds dataset size {}.'.format(
                        label, index, len(dataset)
                    )
                )
            image = dataset[index]
            source_path = dataset.files[index % len(dataset.files)]
            if not isinstance(source_path, str):
                raise RuntimeError('Dataset must use cache=none for manifest hashing.')
            height = image.shape[-2] // 8 * 8
            width = image.shape[-1] // 8 * 8
            if min(height, width) < protocol['kernel_size']:
                raise RuntimeError('Evaluation image is too small for the blur kernel.')
            target_image = image[:, :height, :width].contiguous()
            blur_covariance = covariance_from_parameters(
                entry['sigma_major'], entry['sigma_minor'],
                entry['angle_degrees'], dtype=target_image.dtype,
            )
            hr_kernel = gaussian_kernel(
                blur_covariance, protocol['kernel_size']
            )
            blurred = apply_anisotropic_blur(
                target_image, blur_covariance, protocol['kernel_size']
            ).clamp(0, 1)
            target = target_image.view(3, -1).permute(1, 0) \
                .unsqueeze(0).to(device)
            coord, cell = make_query(height, width, device)
            realized = dict(entry)
            realized.update({
                'filename': os.path.basename(source_path),
                'source_sha256': sha256_file(source_path),
                'height': height,
                'width': width,
            })
            realized_manifest.append(realized)
            for scale_value in scales:
                lr_height = round(height / scale_value)
                lr_width = round(width / scale_value)
                lr = resize_fn(blurred, (lr_height, lr_width)) \
                    .unsqueeze(0).to(device)
                inp = (lr - inp_sub) / inp_div
                scale = torch.tensor([scale_value], device=device)
                model.gen_feat(inp)
                model.set_oracle_covariance_adjustment(None)
                baseline = query_from_features(
                    model, coord, scale, cell, eval_bsize
                )
                baseline = (baseline * gt_div + gt_sub).clamp(0, 1)
                baseline_psnr = psnr(baseline, target)
                for variant in variants:
                    mapped = mapped_internal_covariance(
                        hr_kernel,
                        scale_value,
                        variant['mapping'],
                        renderer_units,
                    ) * variant['alpha']
                    model.set_oracle_covariance_adjustment(
                        mapped, variant['mode'], min_eigenvalue=1e-4
                    )
                    candidate = query_from_features(
                        model, coord, scale, cell, eval_bsize
                    )
                    candidate = (candidate * gt_div + gt_sub).clamp(0, 1)
                    candidate_psnr = psnr(candidate, target)
                    diagnostic = model.last_oracle_covariance_diagnostics
                    if diagnostic is None:
                        raise RuntimeError('Oracle covariance diagnostics are missing.')
                    records[variant['name']].append({
                        'dataset_index': index,
                        'bucket': entry['bucket'],
                        'angle_degrees': entry['angle_degrees'],
                        'scale': float(scale_value),
                        'baseline_psnr': baseline_psnr,
                        'candidate_psnr': candidate_psnr,
                        'delta': candidate_psnr - baseline_psnr,
                        'clamp_fraction': diagnostic['clamp_fraction'].item(),
                        'minimum_eigenvalue': diagnostic['minimum_eigenvalue'].item(),
                        'maximum_eigenvalue': diagnostic['maximum_eigenvalue'].item(),
                    })
                model.set_oracle_covariance_adjustment(None)
            if (entry_index + 1) % 3 == 0 or entry_index + 1 == len(entries):
                print('{} progress: {}/{} images'.format(
                    label, entry_index + 1, len(entries)), flush=True)
    return records, realized_manifest, renderer_units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--helen-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--eval-bsize', type=int, default=20000)
    args = parser.parse_args()
    if args.eval_bsize <= 0:
        raise ValueError('eval-bsize must be positive.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(args.protocol, 'r', encoding='utf-8') as file:
        protocol = json.load(file)
    device = torch.device(args.device)
    model = load_model(args.baseline_checkpoint, config['seed'], device)
    val_spec = config['val_dataset']['dataset']
    celeba = make_dataset(val_spec)
    helen = make_dataset(val_spec, args.helen_root)
    variants = oracle_variants(protocol['alphas'])
    maximum_clamp = protocol['maximum_clamp_fraction']
    gated_scales = protocol['sweep_scales']

    sweep_records, sweep_manifest, renderer_units = evaluate_split(
        model,
        celeba,
        expand_manifest(protocol, 'sweep'),
        protocol['sweep_scales'],
        variants,
        config,
        protocol,
        device,
        args.eval_bsize,
        'SWEEP',
    )
    sweep_summaries = {}
    eligible = []
    for variant in variants:
        name = variant['name']
        summary = summarize(
            sweep_records[name], gated_scales, maximum_clamp
        )
        sweep_summaries[name] = summary
        print_summary('SWEEP {}'.format(name), summary)
        if summary['passes_gate']:
            eligible.append(variant)

    result = {
        'schema_version': 1,
        'protocol': protocol,
        'renderer_units_per_pixel_variance': renderer_units,
        'sweep_manifest': sweep_manifest,
        'sweep_summaries': sweep_summaries,
        'selected_variant': None,
        'celeba_summary': None,
        'helen_summary': None,
        'status': 'sweep_failed',
    }
    if eligible:
        selected = max(
            eligible,
            key=lambda variant: sweep_summaries[variant['name']]['overall_delta'],
        )
        result['selected_variant'] = selected
        print('SELECTED ORACLE VARIANT:', selected['name'])
        celeba_records, celeba_manifest, _ = evaluate_split(
            model,
            celeba,
            expand_manifest(protocol, 'celeba'),
            protocol['confirmation_scales'],
            [selected],
            config,
            protocol,
            device,
            args.eval_bsize,
            'CELEBA',
        )
        celeba_summary = summarize(
            celeba_records[selected['name']], gated_scales, maximum_clamp
        )
        result['celeba_manifest'] = celeba_manifest
        result['celeba_summary'] = celeba_summary
        print_summary('CELEBA {}'.format(selected['name']), celeba_summary)
        result['status'] = 'celeba_failed'
        if celeba_summary['passes_gate']:
            helen_records, helen_manifest, _ = evaluate_split(
                model,
                helen,
                expand_manifest(protocol, 'helen'),
                protocol['confirmation_scales'],
                [selected],
                config,
                protocol,
                device,
                args.eval_bsize,
                'HELEN',
            )
            helen_summary = summarize(
                helen_records[selected['name']], gated_scales, maximum_clamp
            )
            result['helen_manifest'] = helen_manifest
            result['helen_summary'] = helen_summary
            print_summary('HELEN {}'.format(selected['name']), helen_summary)
            result['status'] = (
                'passed' if helen_summary['passes_gate'] else 'helen_failed'
            )

    output_directory = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_directory, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, sort_keys=True)
        file.write('\n')
    print('ORACLE RESULT:', result['status'])
    print('ORACLE JSON:', args.output)
    print('BLIND-SPD ORACLE EVALUATION COMPLETED')


if __name__ == '__main__':
    main()
