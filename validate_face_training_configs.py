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


def compare_protocol(
        left, right, left_name, right_name,
        allow_encoder_difference=False):
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
        if allow_encoder_difference and path == ('model', 'args', 'encoder_spec'):
            continue
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
        '--expected-baseline-model',
        default='gaussian-splatter',
        help='Expected registry name for the reference model.',
    )
    parser.add_argument(
        '--expected-model-b',
        default='gaussian-splatter-local-frequency-scale-v2',
        help='Expected registry name for the candidate model.',
    )
    parser.add_argument(
        '--expected-encoder-b',
        default=None,
        help='Allow and verify an intentional candidate encoder change.',
    )
    args = parser.parse_args()

    baseline = read_yaml(args.baseline)
    model_b = read_yaml(args.model_b)
    validate_model(baseline, args.expected_baseline_model)
    validate_model(model_b, args.expected_model_b)
    if args.expected_encoder_b is not None:
        actual_encoder = model_b['model']['args']['encoder_spec']['name']
        if actual_encoder != args.expected_encoder_b:
            raise ValueError(
                'Expected candidate encoder {!r}, found {!r}.'.format(
                    args.expected_encoder_b, actual_encoder
                )
            )
        baseline_encoder_args = baseline['model']['args']['encoder_spec']['args']
        candidate_encoder_args = model_b['model']['args']['encoder_spec']['args']
        if (
                candidate_encoder_args.get('no_upsampling')
                != baseline_encoder_args.get('no_upsampling')):
            raise ValueError(
                'Candidate encoder changed the no_upsampling protocol.'
            )
    compare_protocol(
        baseline,
        model_b,
        args.baseline,
        args.model_b,
        allow_encoder_difference=(args.expected_encoder_b is not None),
    )
    print('CONFIG VALIDATION PASSED')
    print('seed:', baseline['seed'])
    print('epoch_max:', baseline['epoch_max'])
    print('train batch_size:', baseline['train_dataset']['batch_size'])
    print('baseline:', baseline['model']['name'])
    print('model B:', model_b['model']['name'])
    if args.expected_encoder_b is not None:
        print('model B encoder:', args.expected_encoder_b)


if __name__ == '__main__':
    main()
