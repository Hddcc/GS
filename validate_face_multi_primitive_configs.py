import argparse

import yaml


PROTOCOL_KEYS = (
    'deterministic', 'num_workers', 'train_dataset', 'val_dataset',
    'data_norm', 'optimizer', 'epoch_max', 'multi_step_lr', 'epoch_val',
    'epoch_save', 'eval_bsize',
)


def load(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    errors = []
    for key in PROTOCOL_KEYS:
        if baseline[key] != candidate[key]:
            errors.append('protocol differs at {}'.format(key))
    if baseline.get('seed') != 1 or candidate.get('seed') != 1:
        errors.append('the multi-primitive screen is fixed to seed 1')
    if baseline['model']['name'] != 'gaussian-splatter':
        errors.append('baseline registry name is incorrect')
    if candidate['model']['name'] != 'gaussian-splatter-memory-efficient-m4':
        errors.append('candidate registry name is incorrect')
    baseline_args = baseline['model']['args']
    candidate_args = candidate['model']['args']
    for key in ('encoder_spec', 'dec_spec', 'kernel_size'):
        if baseline_args[key] != candidate_args[key]:
            errors.append('shared model argument differs at {}'.format(key))
    expected = {
        'primitive_chunk_size': 7,
        'gaussian_channels': 8,
        'primitive_count': 4,
        'residual_limit': 0.1,
        'offset_residual_limit': 0.25,
        'covariance_log_limit': 0.25,
    }
    for key, value in expected.items():
        if candidate_args.get(key) != value:
            errors.append('{} must equal {}'.format(key, value))
    if candidate['epoch_max'] != 5 or candidate['epoch_save'] != 5:
        errors.append('candidate must use the five-epoch probe')
    if errors:
        raise ValueError('\n'.join(errors))

    print('MULTI-PRIMITIVE CONFIG VALIDATION PASSED')
    print('seed:', candidate['seed'])
    print('epoch_max:', candidate['epoch_max'])
    print('train batch_size:', candidate['train_dataset']['batch_size'])
    print('model:', candidate['model']['name'])
    print('Gaussian/bicubic channels: 8/56')
    print('primitives per LR position: 4')
    print('primitive chunk:', candidate_args['primitive_chunk_size'])


if __name__ == '__main__':
    main()
