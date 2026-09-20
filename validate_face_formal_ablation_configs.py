import argparse

import yaml


PROTOCOL_PATHS = [
    ('seed',),
    ('deterministic',),
    ('num_workers',),
    ('train_dataset', 'dataset'),
    ('train_dataset', 'batch_size'),
    ('train_dataset', 'scale_grouped_batching'),
    ('train_dataset', 'wrapper', 'args', 'inp_size'),
    ('train_dataset', 'wrapper', 'args', 'scale_min'),
    ('train_dataset', 'wrapper', 'args', 'scale_max'),
    ('train_dataset', 'wrapper', 'args', 'augment'),
    ('train_dataset', 'wrapper', 'args', 'sample_q'),
    ('train_dataset', 'wrapper', 'args', 'batch_per_gpu'),
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


def compare(left, right, left_name, right_name):
    errors = []
    for path in PROTOCOL_PATHS + SHARED_MODEL_PATHS:
        try:
            left_value = nested(left, path)
            right_value = nested(right, path)
        except KeyError as exc:
            errors.append('{} missing key {}'.format('.'.join(path), exc))
            continue
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


def validate_frequency(config):
    if config['model']['name'] != 'gaussian-splatter-face-frequency-residual':
        raise ValueError('frequency ablation must use the frequency residual model.')
    wrapper_args = config['train_dataset']['wrapper']['args']
    if 'edge_weight' in wrapper_args:
        raise ValueError('frequency-only ablation must not define edge_weight.')


def validate_edge(config):
    if config['model']['name'] != 'gaussian-splatter':
        raise ValueError('edge-only ablation must use the GaussianSR model.')
    edge_weight = config['train_dataset']['wrapper']['args'].get('edge_weight')
    if edge_weight != 0.25:
        raise ValueError('edge-only ablation must use edge_weight=0.25.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--frequency', required=True)
    parser.add_argument('--edge', required=True)
    args = parser.parse_args()

    baseline = read_yaml(args.baseline)
    frequency = read_yaml(args.frequency)
    edge = read_yaml(args.edge)
    compare(baseline, frequency, args.baseline, args.frequency)
    compare(baseline, edge, args.baseline, args.edge)
    validate_frequency(frequency)
    validate_edge(edge)
    print('FORMAL ABLATION CONFIG VALIDATION PASSED')
    print('seed:', baseline['seed'])
    print('epoch_max:', baseline['epoch_max'])
    print('train batch_size:', baseline['train_dataset']['batch_size'])
    print('frequency-only: direction-aware query residual, edge_weight=0')
    print('edge-only: GaussianSR, edge_weight=0.25')


if __name__ == '__main__':
    main()
