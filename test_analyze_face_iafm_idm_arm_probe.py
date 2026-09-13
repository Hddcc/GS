import os
import subprocess
import sys
import tempfile


BASELINE = [27.0, 27.4, 27.7, 27.9, 28.1]


def write_train(path, values):
    with open(path, 'w', encoding='utf-8') as file:
        for epoch, value in enumerate(values, 1):
            file.write('epoch {}/5, val: psnr={:.4f}\n'.format(epoch, value))


def write_evaluation(path, deltas, mechanism=0.02):
    with open(path, 'w', encoding='utf-8') as file:
        for scale, delta in zip((2, 4, 8), deltas):
            file.write('x{} psnr delta: {:+.8f}\n'.format(scale, delta))
        file.write('information entropy mean: {:.8f}\n'.format(mechanism))
        file.write('IDM spatial std: {:.8f}\n'.format(mechanism))
        for position in (4, 8, 12, 16):
            file.write('ARM {} response mean abs: {:.8f}\n'.format(
                position, mechanism))
            file.write('ARM {} scaled response mean abs: {:.8f}\n'.format(
                position, mechanism))
        file.write('candidate-baseline output response mean abs: {:.8f}\n'.format(
            mechanism))


def run_case(directory, candidate, deltas, expected, mechanism=0.02):
    baseline_log = os.path.join(directory, 'baseline.log')
    candidate_log = os.path.join(directory, 'candidate.log')
    evaluation_log = os.path.join(directory, 'evaluation.log')
    write_train(baseline_log, BASELINE)
    write_train(candidate_log, candidate)
    write_evaluation(evaluation_log, deltas, mechanism)
    result = subprocess.run([
        sys.executable,
        'analyze_face_iafm_idm_arm_probe.py',
        '--baseline-log', baseline_log,
        '--candidate-log', candidate_log,
        '--evaluation-log', evaluation_log,
    ], capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError('Unexpected analyzer status {}:\n{}\n{}'.format(
            result.returncode, result.stdout, result.stderr))


def main():
    with tempfile.TemporaryDirectory() as directory:
        run_case(
            directory,
            [value + 0.02 for value in BASELINE],
            [0.03, -0.01, 0.02],
            0,
        )
        run_case(
            directory,
            [value + 0.02 for value in BASELINE],
            [-0.01, -0.02, -0.01],
            2,
        )
        run_case(
            directory,
            [value + 0.02 for value in BASELINE],
            [0.03, 0.02, 0.01],
            2,
            mechanism=1e-6,
        )
    print('IAFM IDM+ARM ANALYZER TEST PASSED')


if __name__ == '__main__':
    main()
