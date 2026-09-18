import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    if args.epochs < 4:
        raise ValueError('At least 4 epochs are needed for both training stages.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if checkpoint['model']['name'] != 'gaussian-splatter' or checkpoint.get('seed') != 1:
        raise ValueError('Expected the seed-1 gaussian-splatter formal baseline.')
    source_config_path = checkpoint_path.parent / 'config.yaml'
    with source_config_path.open(encoding='utf-8') as file:
        source_config = yaml.safe_load(file)
    if source_config['epoch_max'] < 1350:
        raise ValueError('This probe requires the formal 1350-epoch baseline run.')
    with Path('configs/train/face/train_face_gaussian_baseline_seed1.yaml').open(
            encoding='utf-8') as file:
        template = yaml.safe_load(file)
    if source_config['data_norm'] != template['data_norm']:
        raise ValueError('Baseline normalization differs from the probe protocol.')
    template.update(epoch_max=args.epochs, epoch_val=1, epoch_save=5,
                    num_workers=4, multi_step_lr=None)
    template.pop('resume', None)
    template['optimizer'] = {'name': 'adam', 'args': {'lr': 1e-6}}
    template['baseline_adaptation'] = {
        'checkpoint': str(checkpoint_path), 'backbone_lr': 1e-6,
        'residual_lr': 1e-4, 'freeze_epochs': 3,
    }
    for split in ('train', 'val'):
        path = root / 'Dataset/FACE/CelebA' / split / 'HR'
        if not path.is_dir():
            raise FileNotFoundError(path)
        template[split + '_dataset']['dataset']['args']['root_path'] = str(path)
    template['train_dataset']['wrapper']['args'].update(scale_min=2, scale_max=4)
    variants = ('control', 'residual', 'gated')
    for variant in variants:
        config = copy.deepcopy(template)
        config['model'] = {key: copy.deepcopy(value) for key, value in
                           checkpoint['model'].items() if key != 'sd'}
        if variant != 'control':
            config['model']['name'] = 'gaussian-splatter-face-scale-gated-frequency'
            config['model']['args'].update(
                use_scale_gate=variant == 'gated', residual_feature_dim=48,
                residual_hidden_dim=64, gate_hidden_dim=32, scale_min=2,
                scale_max=4, max_rgb_residual=0.2,
            )
        path = Path('configs/generated/scale_gated_frequency') / (variant + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, sort_keys=False)
    metadata = {
        'checkpoint': str(checkpoint_path),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'source_config_sha256': hashlib.sha256(source_config_path.read_bytes()).hexdigest(),
        'source_epoch': checkpoint['epoch'], 'fine_tune_epochs': args.epochs,
        'train_scale_range': [2, 4], 'evaluation_scales': [2, 2.5, 3, 3.5, 4],
        'checkpoint_selection': 'fixed final epoch, not grid-selected',
        'status': 'exploratory validation, not formal test results',
    }
    Path('results').mkdir(exist_ok=True)
    Path('results/protocol.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('SCALE-GATED FREQUENCY PROTOCOL PREPARED', flush=True)


if __name__ == '__main__':
    main()
