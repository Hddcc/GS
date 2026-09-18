import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

import evaluate_face_scale_gated_frequency as evaluation
import models
import prepare_face_scale_gated_frequency as preparation
from scale_gated_frequency_training import initialize_from_baseline
from test_face_scale_gated_frequency import model_spec


def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for split in ('train', 'val'):
            folder = temporary / 'Dataset/FACE/CelebA' / split / 'HR'
            folder.mkdir(parents=True)
            for index in range(2):
                pixels = np.random.default_rng(index).integers(
                    0, 256, (40, 48, 3), dtype=np.uint8,
                )
                Image.fromarray(pixels).save(folder / (str(index) + '.png'))
        relative = Path('configs/train/face/train_face_gaussian_baseline_seed1.yaml')
        (temporary / relative).parent.mkdir(parents=True)
        shutil.copyfile(root / relative, temporary / relative)
        source = temporary / 'formal'
        source.mkdir()
        template = yaml.safe_load((root / relative).read_text(encoding='utf-8'))
        (source / 'config.yaml').write_text(yaml.safe_dump({
            'epoch_max': 1350, 'data_norm': template['data_norm'],
        }), encoding='utf-8')
        baseline = models.make(model_spec())
        checkpoint_path = source / 'epoch-best.pth'
        torch.save({'model': dict(model_spec(), sd=baseline.state_dict()),
                    'seed': 1, 'epoch': 1350}, checkpoint_path)
        try:
            os.chdir(temporary)
            with patch.object(sys, 'argv', ['prepare', '--project-root', str(temporary),
                                          '--checkpoint', str(checkpoint_path), '--epochs', '4']):
                preparation.main()
            for variant in ('control', 'residual', 'gated'):
                path = temporary / 'configs/generated/scale_gated_frequency' / (variant + '.yaml')
                config = yaml.safe_load(path.read_text())
                model = models.make(config['model'])
                baseline_source = initialize_from_baseline(model, checkpoint_path, 1)
                folder = temporary / 'save' / ('face_scale_gated_frequency_' + variant + '_seed1')
                folder.mkdir(parents=True)
                spec = copy.deepcopy(config['model'])
                spec['sd'] = model.state_dict()
                torch.save({'model': spec, 'seed': 1, 'epoch': 4,
                            'baseline_source': baseline_source}, folder / 'epoch-last.pth')
            with patch.object(sys, 'argv', ['evaluate', '--samples', '2', '--device', 'cpu', '--chunk', '701']):
                evaluation.main()
            report = json.loads(Path('results/scale_gated_frequency_evaluation.json').read_text())
            assert len(report['per_image']) == 10 and len(report['scales']) == 5
            for result in report['scales'].values():
                assert abs(result['gated_minus_control']['mean']) <= 1e-6
                assert abs(result['gated_minus_residual']['mean']) <= 1e-6
        finally:
            os.chdir(original_cwd)
    print('SCALE-GATED FREQUENCY SYNTHETIC WORKFLOW TEST PASSED')


if __name__ == '__main__':
    main()
