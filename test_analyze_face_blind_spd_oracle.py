import copy

from analyze_face_blind_spd_oracle import analyze


def summary(delta, passes=True):
    return {
        'count': 12,
        'overall_delta': delta,
        'sample_std': 0.01,
        'scale_delta': {'1.5': delta, '2': delta, '4': delta, '8': delta},
        'bucket_delta': {'mild': delta, 'medium': delta},
        'maximum_clamp_fraction': 0.0,
        'minimum_covariance_eigenvalue': 0.1,
        'maximum_covariance_eigenvalue': 3.0,
        'passes_gate': passes,
    }


def base_result():
    selected = {
        'name': 'hr-subtract-a0.25',
        'mapping': 'hr',
        'mode': 'subtract',
        'alpha': 0.25,
    }
    return {
        'protocol': {'maximum_clamp_fraction': 0.1},
        'sweep_summaries': {
            selected['name']: summary(0.03),
            'lr-add-a0.25': summary(-0.02, passes=False),
        },
        'selected_variant': selected,
        'celeba_summary': summary(0.02),
        'helen_summary': summary(0.01),
        'status': 'passed',
    }


def expect(result, expected_pass):
    passed, _ = analyze(result)
    if passed is not expected_pass:
        raise RuntimeError('Unexpected analyzer decision.')


def main():
    expect(base_result(), True)

    sweep_failure = base_result()
    sweep_failure['sweep_summaries'] = {
        'hr-add-a0.25': summary(-0.01, passes=False)
    }
    sweep_failure['selected_variant'] = None
    sweep_failure['celeba_summary'] = None
    sweep_failure['helen_summary'] = None
    sweep_failure['status'] = 'sweep_failed'
    expect(sweep_failure, False)

    celeba_failure = base_result()
    celeba_failure['celeba_summary'] = summary(-0.01, passes=False)
    celeba_failure['helen_summary'] = None
    celeba_failure['status'] = 'celeba_failed'
    expect(celeba_failure, False)

    helen_failure = base_result()
    helen_failure['helen_summary'] = summary(-0.01, passes=False)
    helen_failure['status'] = 'helen_failed'
    expect(helen_failure, False)

    inconsistent = copy.deepcopy(base_result())
    inconsistent['celeba_summary']['passes_gate'] = False
    try:
        analyze(inconsistent)
    except ValueError:
        pass
    else:
        raise RuntimeError('Analyzer accepted an inconsistent stored gate.')
    print('BLIND-SPD ORACLE ANALYZER TEST PASSED')


if __name__ == '__main__':
    main()
