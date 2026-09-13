import argparse
import copy

import yaml


def load(path):
    with open(path, 'r', encoding='utf-8') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--expected-seed', type=int, default=1)
    args = parser.parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    if baseline['seed'] != args.expected_seed or candidate['seed'] != args.expected_seed:
        raise RuntimeError('Config seed mismatch.')
    if candidate['model']['name'] != 'gaussian-splatter':
        raise RuntimeError('Candidate changed the GaussianSR model wrapper.')
    encoder = candidate['model']['args']['encoder_spec']
    if encoder['name'] != 'edsr-face-iafm-idm-arm':
        raise RuntimeError('Candidate does not use the IAFM IDM+ARM encoder.')
    expected = {
        'no_upsampling': True,
        'use_pretrained': False,
        'arm_positions': [4, 8, 12, 16],
        'arm_residual_scale': 0.1,
        'entropy_loss_weight': 1e-4,
    }
    if encoder['args'] != expected:
        raise RuntimeError('Candidate IAFM arguments changed.')

    normalized = copy.deepcopy(candidate)
    normalized['model']['args']['encoder_spec'] = copy.deepcopy(
        baseline['model']['args']['encoder_spec']
    )
    if normalized != baseline:
        raise RuntimeError('Candidate changed fields outside the encoder spec.')
    if candidate['epoch_max'] != 5 or candidate['train_dataset']['batch_size'] != 12:
        raise RuntimeError('Candidate changed the seed-1 probe budget.')
    print('IAFM IDM+ARM CONFIG VALIDATION PASSED')
    print('seed:', candidate['seed'])
    print('epoch_max:', candidate['epoch_max'])
    print('train batch_size:', candidate['train_dataset']['batch_size'])
    print('model:', candidate['model']['name'])
    print('encoder:', encoder['name'])
    print('ARM positions:', ', '.join(map(str, encoder['args']['arm_positions'])))
    print('entropy weight:', encoder['args']['entropy_loss_weight'])


if __name__ == '__main__':
    main()
