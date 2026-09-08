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


def read_candidate(raw_candidate, baseline):
    channels_text, candidate_log, evaluation_log = raw_candidate
    channels = int(channels_text)
    candidate = read_epochs(candidate_log)
    with open(evaluation_log, 'r', encoding='utf-8') as file:
        evaluation = file.read()
    reported_channels = int(read_metric(evaluation, 'Gaussian channels'))
    if reported_channels != channels:
        raise RuntimeError('Evaluation reports the wrong channel budget.')
    scale_deltas = [
        read_metric(evaluation, 'x{} psnr delta'.format(scale))
        for scale in (2, 4, 8)
    ]
    return {
        'channels': channels,
        'mean_delta': statistics.mean(candidate) - statistics.mean(baseline),
        'final_delta': candidate[-1] - baseline[-1],
        'scale_deltas': scale_deltas,
        'heldout_mean': statistics.mean(scale_deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument(
        '--candidate', nargs=3, action='append', required=True,
        metavar=('CHANNELS', 'TRAIN_LOG', 'EVALUATION_LOG'),
    )
    parser.add_argument('--train-mean-floor', type=float, default=-0.030)
    parser.add_argument('--train-final-floor', type=float, default=-0.050)
    parser.add_argument('--heldout-mean-floor', type=float, default=0.0)
    parser.add_argument('--individual-floor', type=float, default=-0.100)
    args = parser.parse_args()

    baseline = read_epochs(args.baseline_log)
    results = [
        read_candidate(candidate, baseline)
        for candidate in args.candidate
    ]
    channels = [result['channels'] for result in results]
    if sorted(channels) != [16, 32, 64] or len(set(channels)) != 3:
        raise RuntimeError('Budget screen requires exactly Cg=16, 32, and 64.')
    results.sort(key=lambda result: result['channels'])

    print('Cg | train_mean | train_final | x2 | x4 | x8 | heldout_mean')
    for result in results:
        x2, x4, x8 = result['scale_deltas']
        print(
            '{} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | '
            '{:+.4f}'.format(
                result['channels'], result['mean_delta'],
                result['final_delta'], x2, x4, x8,
                result['heldout_mean'],
            )
        )

    ranked = sorted(
        results,
        key=lambda result: (-result['heldout_mean'], result['channels']),
    )
    best = ranked[0]
    passed = best['mean_delta'] >= args.train_mean_floor - 1e-12
    passed = (
        passed
        and best['final_delta'] >= args.train_final_floor - 1e-12
        and best['heldout_mean'] > args.heldout_mean_floor
        and min(best['scale_deltas']) >= args.individual_floor - 1e-12
    )
    print('ranking by held-out mean: {}'.format(' > '.join(
        'Cg{}({:+.4f})'.format(result['channels'], result['heldout_mean'])
        for result in ranked
    )))
    print('selected Gaussian channels:', best['channels'])
    print('selected held-out mean delta: {:+.4f} dB'.format(
        best['heldout_mean']
    ))
    print(
        'seed-1 gate: selected train mean/final >= {:+.3f}/{:+.3f}, '
        'held-out mean > {:+.3f}, each scale >= {:+.3f}'.format(
            args.train_mean_floor, args.train_final_floor,
            args.heldout_mean_floor, args.individual_floor,
        )
    )
    print('SELECTED_GAUSSIAN_CHANNELS={}'.format(best['channels']))
    print('GAUSSIAN CHANNEL-BUDGET SEED1 SCREEN {}'.format(
        'PASSED' if passed else 'FAILED'
    ))
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
