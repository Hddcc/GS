import argparse
import copy
from pathlib import Path

import yaml


MODEL_NAME = 'gaussian-splatter-local-frequency-scale'
MODEL_ARGS = {
    'frequency_hidden_dim': 32,
    'max_log_sigma_delta': 0.35,
    'max_rho_delta': 0.25,
    'max_opacity_logit_delta': 1.0,
}


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(
            '{} must be {!r}, but the frozen config contains {!r}.'.format(
                label, expected, actual
            )
        )


def validate_baseline(config):
    train = config['train_dataset']
    train_data = train['dataset']['args']
    train_wrapper = train['wrapper']['args']
    val = config['val_dataset']
    val_wrapper = val['wrapper']['args']

    require_equal(train_data.get('repeat'), 1, 'train repeat')
    require_equal(train_wrapper.get('inp_size'), 16, 'train inp_size')
    require_equal(train_wrapper.get('scale_min'), 1.5, 'train scale_min')
    require_equal(train_wrapper.get('scale_max'), 8, 'train scale_max')
    require_equal(train_wrapper.get('sample_q'), 512, 'train sample_q')
    require_equal(train_wrapper.get('batch_per_gpu'), 4, 'train batch_per_gpu')
    require_equal(train.get('batch_size'), 12, 'train batch_size')
    require_equal(val_wrapper.get('scale_min'), 4, 'validation scale_min')
    require_equal(val_wrapper.get('scale_max'), 4, 'validation scale_max')
    require_equal(val.get('batch_size'), 1, 'validation batch_size')
    require_equal(config.get('epoch_max'), 1350, 'epoch_max')
    if config.get('resume') is not None:
        raise ValueError('The frozen baseline config must not contain resume.')


def write_yaml(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--formal-output', required=True)
    parser.add_argument('--debug-output', required=True)
    args = parser.parse_args()

    with open(args.baseline, 'r', encoding='utf-8') as f:
        baseline = yaml.load(f, Loader=yaml.FullLoader)
    validate_baseline(baseline)

    formal = copy.deepcopy(baseline)
    formal['model']['name'] = MODEL_NAME
    formal['model']['args'].update(MODEL_ARGS)
    formal.pop('resume', None)

    debug = copy.deepcopy(formal)
    debug['train_dataset']['dataset']['args']['first_k'] = 120
    debug['val_dataset']['dataset']['args']['first_k'] = 10
    debug['epoch_max'] = 5
    debug['multi_step_lr'] = {'milestones': [2, 4], 'gamma': 0.5}
    debug['epoch_val'] = 1
    debug['epoch_save'] = 1

    write_yaml(args.formal_output, formal)
    write_yaml(args.debug_output, debug)
    print('baseline config validated')
    print('formal config:', args.formal_output)
    print('debug config:', args.debug_output)
    print('model:', MODEL_NAME)


if __name__ == '__main__':
    main()
