import argparse
import json
import math


GATED_SCALES = ('2', '4', '8')


def validate_summary(summary, maximum_clamp_fraction):
    required = [summary['overall_delta'], summary['sample_std'],
                summary['maximum_clamp_fraction'],
                summary['minimum_covariance_eigenvalue'],
                summary['maximum_covariance_eigenvalue']]
    required.extend(summary['scale_delta'].values())
    required.extend(summary['bucket_delta'].values())
    if not all(math.isfinite(float(value)) for value in required):
        raise ValueError('Oracle summary contains a non-finite metric.')
    expected_pass = bool(
        summary['overall_delta'] > 0
        and summary['bucket_delta']['mild'] > 0
        and summary['bucket_delta']['medium'] > 0
        and all(summary['scale_delta'][scale] >= 0 for scale in GATED_SCALES)
        and summary['maximum_clamp_fraction'] <= maximum_clamp_fraction
        and summary['minimum_covariance_eigenvalue'] > 0
    )
    if summary['passes_gate'] is not expected_pass:
        raise ValueError('Stored oracle gate does not match the fixed criterion.')
    return expected_pass


def analyze(result):
    maximum_clamp = result['protocol']['maximum_clamp_fraction']
    eligible = []
    for name, summary in result['sweep_summaries'].items():
        if validate_summary(summary, maximum_clamp):
            eligible.append(name)
    selected = result['selected_variant']
    if not eligible:
        if selected is not None or result['status'] != 'sweep_failed':
            raise ValueError('Failed sweep has inconsistent selection state.')
        return False, 'BLIND-SPD ORACLE SWEEP FAILED; SUGGESTION D STOPPED'
    if selected is None or selected['name'] not in eligible:
        raise ValueError('Selected oracle variant did not pass the sweep gate.')
    ranked = max(
        eligible,
        key=lambda name: result['sweep_summaries'][name]['overall_delta'],
    )
    if selected['name'] != ranked:
        raise ValueError('Selected oracle variant is not the top eligible variant.')
    celeba_passed = validate_summary(result['celeba_summary'], maximum_clamp)
    if not celeba_passed:
        if result['status'] != 'celeba_failed':
            raise ValueError('CelebA failure has inconsistent status.')
        return False, 'BLIND-SPD ORACLE CELEBA CONFIRMATION FAILED; SUGGESTION D STOPPED'
    helen_passed = validate_summary(result['helen_summary'], maximum_clamp)
    if not helen_passed:
        if result['status'] != 'helen_failed':
            raise ValueError('Helen failure has inconsistent status.')
        return False, 'BLIND-SPD ORACLE HELEN CONFIRMATION FAILED; SUGGESTION D STOPPED'
    if result['status'] != 'passed':
        raise ValueError('Passing oracle result has inconsistent status.')
    return True, 'BLIND-SPD ORACLE ADMISSION PASSED'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    with open(args.result, 'r', encoding='utf-8') as file:
        result = json.load(file)
    passed, message = analyze(result)
    print('selected variant:',
          None if result['selected_variant'] is None
          else result['selected_variant']['name'])
    print(message)
    raise SystemExit(0 if passed else 2)


if __name__ == '__main__':
    main()
