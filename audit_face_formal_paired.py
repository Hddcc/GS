"""Audit fixed-scale, paired formal face-SR metrics without rerunning models."""

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


ALL_SCALES = (1.5, 2.0, 2.5, 3.5, 4.0, 5.5, 7.5, 8.0)
FOCUS_SCALES = (2.0, 2.5, 3.5, 4.0)
METRICS = ('psnr_y', 'ssim_y', 'lpips')
ROLES = ('baseline', 'candidate')
DATASETS = ('CelebA', 'Helen')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def discover(directory, role, dataset):
    role_terms = ('baseline',) if role == 'baseline' else ('candidate', 'proposed')
    matches = [path for path in directory.rglob('*_per_image.csv')
               if dataset.lower() in path.stem.lower()
               and any(term in path.stem.lower() for term in role_terms)]
    if len(matches) != 1:
        raise ValueError('Expected exactly one {} {} per-image CSV in {}; found {}. '
                         'Supply explicit --{}-{} path.'.format(
                             role, dataset, directory, [str(path) for path in matches],
                             role, dataset.lower()))
    return matches[0]


def read_rows(path, dataset, expected_images):
    required = {'method', 'dataset', 'image', 'scale', 'crop_border', *METRICS}
    records = {}
    scales = set()
    images_by_scale = {}
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError('Invalid formal per-image CSV columns: {}'.format(path))
        for row in reader:
            if row['method'] != 'model' or row['dataset'].casefold() != dataset.casefold():
                raise ValueError('Method/dataset mismatch in {}'.format(path))
            scale = float(row['scale'])
            if not math.isfinite(scale) or scale not in ALL_SCALES:
                raise ValueError('Unexpected scale {} in {}'.format(scale, path))
            border = int(row['crop_border'])
            if border != math.ceil(scale):
                raise ValueError('Border mismatch at scale {} in {}'.format(scale, path))
            name = row['image'].strip()
            if not name:
                raise ValueError('Empty image name in {}'.format(path))
            key = (scale, name)
            if key in records:
                raise ValueError('Duplicate image/scale {} in {}'.format(key, path))
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError('Nonfinite metric at {} in {}'.format(key, path))
            records[key] = values
            scales.add(scale)
            images_by_scale.setdefault(scale, set()).add(name)
    if scales != set(ALL_SCALES):
        raise ValueError('Missing/extra scales in {}: {}'.format(path, sorted(scales)))
    image_set = images_by_scale[ALL_SCALES[0]]
    if len(image_set) != expected_images or any(
            images_by_scale[scale] != image_set for scale in ALL_SCALES):
        raise ValueError('Incomplete or inconsistent per-scale image set in {}'.format(path))
    return records


def paired_summary(deltas, seed, draws, higher_is_better=True):
    if not deltas:
        raise ValueError('No paired values to summarize.')
    rng = random.Random(seed)
    size = len(deltas)
    samples = [statistics.mean(deltas[rng.randrange(size)] for _ in range(size))
               for _ in range(draws)]
    samples.sort()
    lower = samples[int(0.025 * (draws - 1))]
    upper = samples[int(0.975 * (draws - 1))]
    return {'mean': statistics.mean(deltas),
            'ci95_image_bootstrap': [lower, upper],
            'improved_fraction': sum(
                delta > 0 if higher_is_better else delta < 0 for delta in deltas
            ) / size,
            'n_images': size}


def summarize(dataset, baseline, candidate, draws):
    if baseline.keys() != candidate.keys():
        missing = sorted(baseline.keys() - candidate.keys())[:3]
        extra = sorted(candidate.keys() - baseline.keys())[:3]
        raise ValueError('{} CSVs have unpaired image/scale keys: missing {}, extra {}'.format(
            dataset, missing, extra))
    scale_results = {}
    for scale in ALL_SCALES:
        keys = sorted(key for key in baseline if key[0] == scale)
        details = {}
        for metric in METRICS:
            deltas = [candidate[key][metric] - baseline[key][metric] for key in keys]
            details[metric] = paired_summary(
                deltas, int(scale * 1000) + len(keys), draws,
                higher_is_better=metric != 'lpips',
            )
            details[metric]['baseline_mean'] = statistics.mean(baseline[key][metric] for key in keys)
            details[metric]['candidate_mean'] = statistics.mean(candidate[key][metric] for key in keys)
        scale_results[str(scale)] = details
    focus = {}
    image_names = sorted(key[1] for key in baseline if key[0] == FOCUS_SCALES[0])
    for metric in METRICS:
        per_image_means = [statistics.mean(
            candidate[scale, name][metric] - baseline[scale, name][metric]
            for scale in FOCUS_SCALES) for name in image_names]
        focus[metric] = paired_summary(
            per_image_means, 10000 + len(image_names), draws,
            higher_is_better=metric != 'lpips',
        )
    return {'dataset': dataset, 'scales': scale_results, 'focus_x2_to_x4': focus}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-dir', type=Path, default=Path('results/face_formal_v2protocol/metrics'))
    parser.add_argument('--baseline-celeba', type=Path)
    parser.add_argument('--candidate-celeba', type=Path)
    parser.add_argument('--baseline-helen', type=Path)
    parser.add_argument('--candidate-helen', type=Path)
    parser.add_argument('--celeba-count', type=int, default=1000)
    parser.add_argument('--helen-count', type=int, default=50)
    parser.add_argument('--draws', type=int, default=5000)
    parser.add_argument('--output', type=Path,
                        default=Path('results/face_formal_v2protocol/paired_audit_20260919.json'))
    args = parser.parse_args(argv)
    if args.draws < 100 or args.celeba_count <= 0 or args.helen_count <= 0:
        parser.error('draws >= 100 and positive dataset sizes are required')
    sources = {}
    reports = {}
    for dataset in DATASETS:
        values = {}
        for role in ROLES:
            path = getattr(args, '{}_{}'.format(role, dataset.lower()))
            path = path if path is not None else discover(args.metrics_dir, role, dataset)
            if not path.is_file():
                raise FileNotFoundError(path)
            sources['{}_{}'.format(role, dataset.lower())] = {
                'path': str(path.resolve()), 'sha256': sha256(path)}
            values[role] = read_rows(path, dataset, getattr(args, dataset.lower() + '_count'))
        reports[dataset] = summarize(dataset, values['baseline'], values['candidate'], args.draws)
    report = {'status': 'retrospective formal per-image audit, seed 1 only',
              'metric_direction': {'psnr_y': 'higher', 'ssim_y': 'higher', 'lpips': 'lower'},
              'focus_scales': list(FOCUS_SCALES), 'all_reported_scales': list(ALL_SCALES),
              'ci_note': 'Paired image bootstrap; does not quantify training-seed uncertainty.',
              'bootstrap_draws': args.draws, 'sources': sources, 'results': reports}
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite previous audit: {}'.format(args.output))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('dataset scale metric delta 95% image CI improved_image_fraction')
    for dataset in DATASETS:
        for scale in ALL_SCALES:
            for metric in METRICS:
                values = reports[dataset]['scales'][str(scale)][metric]
                print('{} x{:g} {} {:+.6f} [{:+.6f}, {:+.6f}] {:.3f}'.format(
                    dataset, scale, metric, values['mean'], *values['ci95_image_bootstrap'],
                    values['improved_fraction']))
        for metric in METRICS:
            values = reports[dataset]['focus_x2_to_x4'][metric]
            print('{} focus x2/2.5/3.5/4 {} {:+.6f} [{:+.6f}, {:+.6f}] {:.3f}'.format(
                dataset, metric, values['mean'], *values['ci95_image_bootstrap'],
                values['improved_fraction']))
    print('AUDIT COMPLETED; JSON={}'.format(args.output))


if __name__ == '__main__':
    main()
