import argparse
import math
import re
import statistics


EPOCH_PATTERN = re.compile(
    r'epoch\s+(\d+)/(\d+).*?val:\s*psnr=([-+0-9.eE]+)'
)


def read_epochs(path):
    values = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            match = EPOCH_PATTERN.search(line)
            if match is not None:
                epoch = int(match.group(1))
                if epoch in values:
                    raise RuntimeError('{} repeats epoch {}.'.format(path, epoch))
                values[epoch] = float(match.group(3))
    if set(values) != set(range(1, 6)):
        raise RuntimeError('{} must contain exactly epochs 1 through 5.'.format(path))
    return [values[epoch] for epoch in range(1, 6)]


def read_metric(text, label):
    match = re.search(
        r'^{}:\s*([-+0-9.eE]+)\s*$'.format(re.escape(label)),
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise RuntimeError('Missing evaluation metric: {}.'.format(label))
    value = float(match.group(1))
    if not math.isfinite(value):
        raise RuntimeError('Evaluation metric is not finite: {}.'.format(label))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument('--candidate-log', required=True)
    parser.add_argument('--evaluation-log', required=True)
    parser.add_argument('--train-mean-floor', type=float, default=-0.030)
    parser.add_argument('--train-final-floor', type=float, default=-0.050)
    parser.add_argument('--heldout-mean-floor', type=float, default=0.0)
    parser.add_argument('--individual-floor', type=float, default=-0.100)
    parser.add_argument('--minimum-response', type=float, default=1e-4)
    parser.add_argument('--minimum-diversity', type=float, default=0.01)
    parser.add_argument('--minimum-opacity', type=float, default=1e-3)
    args = parser.parse_args()

    baseline = read_epochs(args.baseline_log)
    candidate = read_epochs(args.candidate_log)
    with open(args.evaluation_log, 'r', encoding='utf-8') as file:
        evaluation = file.read()
    scale_deltas = [
        read_metric(evaluation, 'x{} psnr delta'.format(scale))
        for scale in (2, 4, 8)
    ]
    branch_responses = [
        read_metric(
            evaluation, 'primitive {} color response mean abs'.format(branch)
        )
        for branch in range(1, 5)
    ]
    train_mean = statistics.mean(candidate) - statistics.mean(baseline)
    train_final = candidate[-1] - baseline[-1]
    heldout_mean = statistics.mean(scale_deltas)
    diversity = read_metric(evaluation, 'normalized primitive diversity')
    separation = read_metric(evaluation, 'minimum primitive center distance')
    opacity = read_metric(evaluation, 'minimum opacity weight')
    renderer_response = read_metric(
        evaluation, 'unclamped renderer response mean abs'
    )
    mechanism_passed = (
        min(branch_responses) > args.minimum_response
        and diversity > args.minimum_diversity
        and separation > 0.1
        and opacity > args.minimum_opacity
        and renderer_response > args.minimum_response
    )
    passed = (
        train_mean >= args.train_mean_floor - 1e-12
        and train_final >= args.train_final_floor - 1e-12
        and heldout_mean > args.heldout_mean_floor
        and min(scale_deltas) >= args.individual_floor - 1e-12
        and mechanism_passed
    )
    x2, x4, x8 = scale_deltas
    print(
        'seed | train_mean | train_final | x2 | x4 | x8 | heldout_mean | '
        'branch1 | branch2 | branch3 | branch4 | diversity | separation | '
        'opacity | renderer'
    )
    print(
        '1 | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | '
        '{:+.4f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | '
        '{:.6f} | {:.6f} | {:.6f}'.format(
            train_mean, train_final, x2, x4, x8, heldout_mean,
            *branch_responses, diversity, separation, opacity,
            renderer_response,
        )
    )
    mechanism_passed = (
        min(branch_responses) > args.minimum_response
        and diversity > args.minimum_diversity
        and separation > 0.1
        and opacity > args.minimum_opacity
        and renderer_response > args.minimum_response
    )
    passed = (
        train_mean >= args.train_mean_floor - 1e-12
        and train_final >= args.train_final_floor - 1e-12
        and heldout_mean > args.heldout_mean_floor
        and min(scale_deltas) >= args.individual_floor - 1e-12
        and mechanism_passed
    )
    print(
        'seed-1 gate: train mean/final >= {:+.3f}/{:+.3f}, held-out '
        'mean > {:+.3f}, each scale >= {:+.3f}'.format(
            args.train_mean_floor, args.train_final_floor,
            args.heldout_mean_floor, args.individual_floor,
        )
    )
    print(
        'mechanism gate: every color response/renderer > {:.1e}, '
        'diversity > {:.3f}, separation > 0.1, opacity > {:.1e}'.format(
            args.minimum_response,
            args.minimum_diversity,
            args.minimum_opacity,
        )
    )
    print('MULTI-PRIMITIVE SEED1 SCREEN {}'.format(
        'PASSED' if passed else 'FAILED'
    ))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
