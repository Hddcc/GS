import json
import os
import subprocess
import sys
import tempfile


SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'analyze_face_gsasr_zero_train.py',
)
ENCODER = 'a61351983470a6bbda0691980cfcba0d3b75a07305c2a14000ce663a49e54847'
DECODER = 'b96999739e6d5f6a81ef9aa432ee2350ed21b205bc0e6b98e3a40613f6898d87'
BASELINE_MEANS = {2: 34.08013660, 4: 28.04536727, 8: 23.67010000}


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(value, file)


def record(index, scale, value):
    return {
        'index': index,
        'filename': 'image_{:03d}.png'.format(index),
        'source_sha256': 'source-{:03d}'.format(index),
        'height': 128,
        'width': 128,
        'scale': scale,
        'lr_height': 128 // scale,
        'lr_width': 128 // scale,
        'lr_sha256': 'lr-{}-{:03d}'.format(scale, index),
        'psnr': value,
    }


def make_files(root, deltas, tamper_lr=False):
    baseline_records = []
    for scale in (2, 4, 8):
        for index in range(2):
            baseline_records.append(record(index, scale, BASELINE_MEANS[scale]))
    baseline = {
        'protocol': 'face-gsasr-zero-train-v1',
        'resize_protocol': 'torchvision-pil-bicubic-uint8-v1',
        'samples': 2,
        'scales': [2, 4, 8],
        'records': baseline_records,
    }
    baseline_path = os.path.join(root, 'baseline.json')
    write_json(baseline_path, baseline)
    candidates = []
    for scale in (2, 4, 8):
        records = [
            record(index, scale, BASELINE_MEANS[scale] + deltas[scale])
            for index in range(2)
        ]
        if tamper_lr and scale == 4:
            records[0]['lr_sha256'] = 'tampered'
        candidate = {
            'protocol': 'face-gsasr-zero-train-v1',
            'resize_protocol': 'torchvision-pil-bicubic-uint8-v1',
            'official_commit': '9d2eb64a51303ce7a22fd672197185488b38e10a',
            'encoder_sha256': ENCODER,
            'decoder_sha256': DECODER,
            'scale': scale,
            'samples': 2,
            'repeat_max_error': 0.0,
            'records': records,
        }
        path = os.path.join(root, 'candidate_x{}.json'.format(scale))
        write_json(path, candidate)
        candidates.append(path)
    return baseline_path, candidates


def run_case(root, deltas, tamper_lr=False):
    baseline, candidates = make_files(root, deltas, tamper_lr=tamper_lr)
    return subprocess.run(
        [
            sys.executable,
            SCRIPT,
            '--baseline-json', baseline,
            '--candidate-json', *candidates,
            '--expected-samples', '2',
        ],
        capture_output=True,
        text=True,
    )


def main():
    with tempfile.TemporaryDirectory() as root:
        passed = run_case(root, {2: 0.01, 4: 0.02, 8: 0.03})
        if passed.returncode != 0 \
                or 'GSASR ZERO-TRAIN ADMISSION PASSED' not in passed.stdout:
            raise RuntimeError('Analyzer rejected the positive fixture.')
    with tempfile.TemporaryDirectory() as root:
        failed = run_case(root, {2: -0.10, 4: 0.01, 8: 0.01})
        if failed.returncode != 2 \
                or 'GSASR ZERO-TRAIN ADMISSION FAILED' not in failed.stdout:
            raise RuntimeError('Analyzer accepted the negative fixture.')
    with tempfile.TemporaryDirectory() as root:
        tampered = run_case(
            root, {2: 0.01, 4: 0.02, 8: 0.03}, tamper_lr=True
        )
        if tampered.returncode == 0 \
                or 'Candidate metadata mismatch.' not in tampered.stderr:
            raise RuntimeError('Analyzer accepted mismatched LR metadata.')
    print('GSASR ZERO-TRAIN ANALYZER TEST PASSED')


if __name__ == '__main__':
    main()
