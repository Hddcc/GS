import argparse

import yaml


PROTOCOL_PATHS = [
    ('seed',),
    ('deterministic',),
    ('num_workers',),
    ('train_dataset', 'dataset'),
    ('train_dataset', 'wrapper'),
    ('train_dataset', 'batch_size'),
    ('train_dataset', 'scale_grouped_batching'),
    ('val_dataset',),
    ('data_norm',),
    ('optimizer',),
    ('epoch_max',),
    ('multi_step_lr',),
    ('epoch_val',),
    ('epoch_save',),
    ('eval_bsize',),
]

SHARED_MODEL_PATHS = [
    ('model', 'args', 'encoder_spec'),
    ('model', 'args', 'dec_spec'),
    ('model', 'args', 'kernel_size'),
]


def read_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def nested(config, path):
    value = config
    for key in path:
        value = value[key]
    return value


def compare_protocol(left, right, left_name, right_name):
    errors = []
    for path in PROTOCOL_PATHS:
        left_value = nested(left, path)
        right_value = nested(right, path)
        if left_value != right_value:
            errors.append(
                '{} differs: {!r} != {!r}'.format(
                    '.'.join(path), left_value, right_value
                )
            )
    if errors:
        raise ValueError(
            '{} and {} are not protocol-equivalent:\n{}'.format(
                left_name, right_name, '\n'.join(errors)
            )
        )

    for path in SHARED_MODEL_PATHS:
        left_value = nested(left, path)
        right_value = nested(right, path)
        if left_value != right_value:
            raise ValueError(
                'Shared model field {} differs: {!r} != {!r}'.format(
                    '.'.join(path), left_value, right_value
                )
            )


def validate_model(config, expected_name):
    actual_name = config['model']['name']
    if actual_name != expected_name:
        raise ValueError(
            'Expected model {!r}, found {!r}.'.format(
                expected_name, actual_name
            )
        )
    encoder_args = config['model']['args']['encoder_spec']['args']
    if encoder_args.get('use_pretrained') is not False:
        raise ValueError('use_pretrained must be explicitly false.')
    if config.get('seed') is None:
        raise ValueError('seed must be explicit.')
    if config['train_dataset'].get('scale_grouped_batching') is not True:
        raise ValueError('scale_grouped_batching must be enabled.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--model-b', required=True)
    parser.add_argument(
        '--expected-model-b',
        default='gaussian-splatter-local-frequency-scale-v2',
        help='Expected registry name for the candidate model.',
    )
    args = parser.parse_args()

    baseline = read_yaml(args.baseline)
    model_b = read_yaml(args.model_b)
    validate_model(baseline, 'gaussian-splatter')
    validate_model(model_b, args.expected_model_b)
    compare_protocol(baseline, model_b, args.baseline, args.model_b)
    print('CONFIG VALIDATION PASSED')
    print('seed:', baseline['seed'])
    print('epoch_max:', baseline['epoch_max'])
    print('train batch_size:', baseline['train_dataset']['batch_size'])
    print('baseline:', baseline['model']['name'])
    print('model B:', model_b['model']['name'])


if __name__ == '__main__':
    main()
