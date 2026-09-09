import os
import subprocess
import sys
import tempfile


BASELINE = [27.0, 27.4, 27.7, 27.9, 28.1]


def write_train(path, values):
    with open(path, 'w', encoding='utf-8') as file:
        for epoch, value in enumerate(values, 1):
            file.write('epoch {}/5, val: psnr={:.4f}\n'.format(epoch, value))


def write_evaluation(path, deltas, branch_value=0.02):
    with open(path, 'w', encoding='utf-8') as file:
        for scale, delta in zip((2, 4, 8), deltas):
            file.write('x{} psnr delta: {:+.8f}\n'.format(scale, delta))
        for branch in range(1, 5):
            file.write(
                'primitive {} color response mean abs: {:.8f}\n'.format(
                    branch, branch_value * branch
                )
            )
        file.write('normalized primitive diversity: 0.50000000\n')
        file.write('minimum primitive center distance: 0.40000000\n')
        file.write('minimum opacity weight: 0.20000000\n')
        file.write('unclamped renderer response mean abs: 0.10000000\n')


def run_case(directory, candidate, deltas, expected):
    baseline_log = os.path.join(directory, 'baseline.log')
    candidate_log = os.path.join(directory, 'candidate.log')
    evaluation_log = os.path.join(directory, 'evaluation.log')
    write_train(baseline_log, BASELINE)
    write_train(candidate_log, candidate)
    write_evaluation(evaluation_log, deltas)
    result = subprocess.run([
        sys.executable,
        'analyze_face_multi_primitive_probe.py',
        '--baseline-log', baseline_log,
        '--candidate-log', candidate_log,
        '--evaluation-log', evaluation_log,
    ], capture_output=True, text=True)
    if result.returncode != expected:
        raise RuntimeError(
            'Unexpected analyzer status {}:\n{}\n{}'.format(
                result.returncode, result.stdout, result.stderr
            )
        )


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
    print('MULTI-PRIMITIVE ANALYZER TEST PASSED')


if __name__ == '__main__':
    main()
