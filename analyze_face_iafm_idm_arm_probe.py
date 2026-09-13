import argparse
import math
import re
import statistics


EPOCH_PATTERN = re.compile(r'epoch\s+\d+/\d+.*?val:\s+psnr=([-+0-9.eE]+)')


def read(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


def epoch_psnr(path):
    values = [float(value) for value in EPOCH_PATTERN.findall(read(path))]
    if len(values) != 5 or not all(math.isfinite(value) for value in values):
        raise RuntimeError('{} does not contain five finite PSNR values.'.format(path))
    return values


def extract_one(text, pattern, label):
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError('Expected one {} value; found {}.'.format(
            label, len(matches)))
    value = float(matches[0])
    if not math.isfinite(value):
        raise RuntimeError('{} is not finite.'.format(label))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-log', required=True)
    parser.add_argument('--candidate-log', required=True)
    parser.add_argument('--evaluation-log', required=True)
    args = parser.parse_args()
    baseline = epoch_psnr(args.baseline_log)
    candidate = epoch_psnr(args.candidate_log)
    evaluation = read(args.evaluation_log)
    deltas = [candidate_value - baseline_value for baseline_value, candidate_value
              in zip(baseline, candidate)]
    heldout = {
        scale: extract_one(
            evaluation,
            r'x{} psnr delta:\s*([-+0-9.eE]+)'.format(scale),
            'x{} delta'.format(scale),
        )
        for scale in (2, 4, 8)
    }
    entropy = extract_one(
        evaluation, r'information entropy mean:\s*([-+0-9.eE]+)', 'entropy')
    idm_std = extract_one(
        evaluation, r'IDM spatial std:\s*([-+0-9.eE]+)', 'IDM spatial std')
    arm = [extract_one(
        evaluation,
        r'ARM {} response mean abs:\s*([-+0-9.eE]+)'.format(position),
        'ARM {} response'.format(position),
    ) for position in (4, 8, 12, 16)]
    scaled = [extract_one(
        evaluation,
        r'ARM {} scaled response mean abs:\s*([-+0-9.eE]+)'.format(position),
        'ARM {} scaled response'.format(position),
    ) for position in (4, 8, 12, 16)]
    output_response = extract_one(
        evaluation,
        r'candidate-baseline output response mean abs:\s*([-+0-9.eE]+)',
        'output response',
    )
    train_mean = statistics.mean(deltas)
    train_final = deltas[-1]
    heldout_mean = statistics.mean(heldout.values())
    quality_passed = bool(
        train_mean >= -0.01
        and train_final >= -0.02
        and heldout_mean > 0
        and all(value >= -0.03 for value in heldout.values())
    )
    mechanism_passed = bool(
        entropy > 1e-4
        and idm_std > 1e-4
        and min(arm) > 1e-4
        and min(scaled) > 1e-4
        and output_response > 1e-4
    )
    print('seed | train_mean | train_final | x2 | x4 | x8 | heldout_mean | entropy | idm_std | arm_min | scaled_min | output')
    print('1 | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:+.4f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f}'.format(
        train_mean, train_final, heldout[2], heldout[4], heldout[8],
        heldout_mean, entropy, idm_std, min(arm), min(scaled),
        output_response,
    ))
    print('seed-1 gate: train mean/final >= -0.010/-0.020, held-out mean > +0.000, each held-out scale >= -0.030')
    print('mechanism gate: entropy/IDM/each ARM/scaled/output response > 1.0e-04')
    if quality_passed and mechanism_passed:
        print('IAFM IDM+ARM SEED1 SCREEN PASSED')
        return
    print('IAFM IDM+ARM SEED1 SCREEN FAILED')
    raise SystemExit(2)


if __name__ == '__main__':
    main()
