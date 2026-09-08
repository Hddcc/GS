import os
import subprocess
import sys
import tempfile


def write_epoch_log(path, values):
    with open(path, 'w', encoding='utf-8') as file:
        for epoch, value in enumerate(values, 1):
            file.write(
                'epoch {}/5, train: loss=0.1, val: psnr={:.4f}\n'.format(
                    epoch, value
                )
            )


def write_evaluation(path, channels, deltas):
    with open(path, 'w', encoding='utf-8') as file:
        file.write('Gaussian channels: {}\n'.format(channels))
        for scale, value in zip((2, 4, 8), deltas):
            file.write('x{} psnr delta: {:+.8f}\n'.format(scale, value))


def run_case(directory, candidates):
    baseline = os.path.join(directory, 'baseline.log')
    write_epoch_log(baseline, [27.0, 27.4, 27.7, 27.9, 28.1])
    command = [
        sys.executable,
        'analyze_face_gaussian_channel_budget.py',
        '--baseline-log', baseline,
    ]
    for channels, epoch_delta, scale_deltas in candidates:
        train_log = os.path.join(directory, 'train_{}.log'.format(channels))
        eval_log = os.path.join(directory, 'eval_{}.log'.format(channels))
        write_epoch_log(
            train_log,
            [value + epoch_delta for value in (27.0, 27.4, 27.7, 27.9, 28.1)],
        )
        write_evaluation(eval_log, channels, scale_deltas)
        command.extend([
            '--candidate', str(channels), train_log, eval_log,
        ])
    return subprocess.run(
        command, check=False, capture_output=True, text=True
    )


def main():
    with tempfile.TemporaryDirectory() as directory:
        passed = run_case(directory, [
            (16, 0.005, [-0.01, 0.02, 0.01]),
            (32, 0.010, [-0.02, 0.04, 0.02]),
            (64, -0.005, [-0.01, -0.01, 0.00]),
        ])
        if passed.returncode != 0:
            raise RuntimeError(passed.stdout + passed.stderr)
        if 'SELECTED_GAUSSIAN_CHANNELS=32' not in passed.stdout:
            raise RuntimeError('Passing case selected the wrong budget.')
        if 'SEED1 SCREEN PASSED' not in passed.stdout:
            raise RuntimeError('Synthetic passing case did not pass.')

        failed = run_case(directory, [
            (16, 0.005, [-0.01, -0.01, -0.01]),
            (32, 0.005, [-0.02, 0.00, 0.01]),
            (64, 0.005, [-0.03, 0.01, 0.01]),
        ])
        if failed.returncode != 2:
            raise RuntimeError(failed.stdout + failed.stderr)
        if 'SEED1 SCREEN FAILED' not in failed.stdout:
            raise RuntimeError('Synthetic failing case did not fail.')
    print('GAUSSIAN CHANNEL-BUDGET ANALYZER TEST PASSED')


if __name__ == '__main__':
    main()
