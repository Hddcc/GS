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
    if baseline['model']['name'] != 'gaussian-splatter':
        errors.append('baseline registry name is incorrect')
    if candidate['model']['name'] != 'gaussian-splatter-memory-efficient':
        errors.append('candidate registry name is incorrect')
    baseline_args = baseline['model']['args']
    candidate_args = candidate['model']['args']
    for key in ('encoder_spec', 'dec_spec', 'kernel_size'):
        if baseline_args[key] != candidate_args[key]:
            errors.append('shared model argument differs at {}'.format(key))
    if candidate_args.get('primitive_chunk_size') != 7:
        errors.append('primitive_chunk_size must be 7')
    if candidate_args.get('gaussian_channels') != 8:
        errors.append('gaussian_channels must be 8 in stage 1')
    if errors:
        raise ValueError('\n'.join(errors))
    print('MEMORY-EFFICIENT RASTER CONFIG VALIDATION PASSED')
    print('seed:', candidate['seed'])
    print('epoch_max:', candidate['epoch_max'])
    print('model:', candidate['model']['name'])
    print('primitive chunk:', candidate_args['primitive_chunk_size'])
    print('Gaussian channels:', candidate_args['gaussian_channels'])


if __name__ == '__main__':
    main()
