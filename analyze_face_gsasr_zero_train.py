import argparse
import json
import math
import statistics


EXPECTED_BASELINE = {
    2: 34.08013660,
    4: 28.04536727,
    8: 23.67010000,
}
EXPECTED_ENCODER = 'a61351983470a6bbda0691980cfcba0d3b75a07305c2a14000ce663a49e54847'
EXPECTED_DECODER = 'b96999739e6d5f6a81ef9aa432ee2350ed21b205bc0e6b98e3a40613f6898d87'
EXPECTED_RESIZE_PROTOCOL = 'torchvision-pil-bicubic-uint8-v1'


def load(path):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def keyed(records):
    result = {}
    for record in records:
        key = (record['index'], record['filename'], record['scale'])
        if key in result:
            raise RuntimeError('Duplicate evaluation record: {}'.format(key))
        result[key] = record
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-json', required=True)
    parser.add_argument('--candidate-json', nargs=3, required=True)
    parser.add_argument('--expected-samples', type=int, default=100)
    args = parser.parse_args()
    baseline = load(args.baseline_json)
    if baseline.get('protocol') != 'face-gsasr-zero-train-v1':
        raise RuntimeError('Baseline protocol mismatch.')
    if baseline.get('resize_protocol') != EXPECTED_RESIZE_PROTOCOL:
        raise RuntimeError('Baseline resize protocol mismatch.')
    if baseline.get('samples') != args.expected_samples \
            or baseline.get('scales') != [2, 4, 8]:
        raise RuntimeError('Baseline sample/scale declaration mismatch.')
    baseline_records = keyed(baseline['records'])
    deltas = {}
    candidate_means = {}
    repeat_errors = {}
    for path in args.candidate_json:
        candidate = load(path)
        if candidate.get('protocol') != 'face-gsasr-zero-train-v1' \
                or candidate.get('resize_protocol') != EXPECTED_RESIZE_PROTOCOL:
            raise RuntimeError('Candidate protocol mismatch.')
        scale = int(candidate['scale'])
        if scale not in (2, 4, 8) or scale in deltas:
            raise RuntimeError('Candidate scale set is invalid.')
        if candidate.get('official_commit') != \
                '9d2eb64a51303ce7a22fd672197185488b38e10a':
            raise RuntimeError('Official commit mismatch.')
        if candidate.get('encoder_sha256') != EXPECTED_ENCODER \
                or candidate.get('decoder_sha256') != EXPECTED_DECODER:
            raise RuntimeError('Official checkpoint hash mismatch.')
        if candidate.get('samples') != args.expected_samples:
            raise RuntimeError('Candidate sample declaration mismatch.')
        records = keyed(candidate['records'])
        if len(records) != args.expected_samples:
            raise RuntimeError('Candidate sample count mismatch.')
        differences = []
        candidate_values = []
        for key, record in records.items():
            reference = baseline_records.get(key)
            if reference is None:
                raise RuntimeError('Candidate is not paired to baseline.')
            for field in (
                    'source_sha256', 'height', 'width',
                    'lr_height', 'lr_width', 'lr_sha256'):
                if record[field] != reference[field]:
                    raise RuntimeError('Candidate metadata mismatch.')
            if record['scale'] != scale:
                raise RuntimeError('Candidate record scale mismatch.')
            if not math.isfinite(record['psnr']):
                raise RuntimeError('Candidate PSNR is not finite.')
            differences.append(record['psnr'] - reference['psnr'])
            candidate_values.append(record['psnr'])
        deltas[scale] = statistics.mean(differences)
        candidate_means[scale] = statistics.mean(candidate_values)
        repeat_errors[scale] = float(candidate['repeat_max_error'])
        if not math.isfinite(repeat_errors[scale]):
            raise RuntimeError('Candidate repeat error is not finite.')
    if set(deltas) != {2, 4, 8}:
        raise RuntimeError('Candidate must contain x2, x4, and x8.')
    if len(baseline_records) != args.expected_samples * 3:
        raise RuntimeError('Baseline sample count mismatch.')

    baseline_means = {}
    for scale in (2, 4, 8):
        values = [
            record['psnr'] for record in baseline_records.values()
            if record['scale'] == scale
        ]
        if len(values) != args.expected_samples \
                or not all(math.isfinite(value) for value in values):
            raise RuntimeError('Baseline records are invalid.')
        baseline_means[scale] = statistics.mean(values)
        if abs(baseline_means[scale] - EXPECTED_BASELINE[scale]) > 5e-6:
            raise RuntimeError('Baseline x{} protocol drift: {:.8f}'.format(
                scale, baseline_means[scale]))
    heldout_mean = statistics.mean(deltas.values())
    repeat_max = max(repeat_errors.values())
    passed = bool(
        heldout_mean > 0
        and all(value >= -0.03 for value in deltas.values())
        and repeat_max <= 1e-6
    )
    print('scale | baseline | GSASR | delta | repeat_max')
    for scale in (2, 4, 8):
        print('x{} | {:.8f} | {:.8f} | {:+.8f} | {:.3e}'.format(
            scale, baseline_means[scale], candidate_means[scale],
            deltas[scale], repeat_errors[scale]))
    print('held-out mean delta: {:+.8f}'.format(heldout_mean))
    print('gate: held-out mean > 0, each scale >= -0.030 dB, repeat max <= 1.0e-06')
    if passed:
        print('GSASR ZERO-TRAIN ADMISSION PASSED')
        return
    print('GSASR ZERO-TRAIN ADMISSION FAILED')
    raise SystemExit(2)


if __name__ == '__main__':
    main()
