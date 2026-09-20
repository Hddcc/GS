""" Train GaussianSR
    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import copy
import math
import os
import random

import numpy as np
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr


data_generators = {}


def set_seed(seed, deterministic=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(name, seed, offset):
    generator = torch.Generator()
    generator.manual_seed(seed + offset)
    data_generators[name] = generator
    return generator


class ScaleGroupedBatchSampler:
    """Keep one scale per contiguous per-GPU group in each batch."""

    def __init__(
            self, dataset_size, batch_size, batch_per_gpu,
            scale_min, scale_max, generator, curriculum=None):
        if batch_size % batch_per_gpu != 0:
            raise ValueError(
                'batch_size must be divisible by batch_per_gpu when '
                'scale_grouped_batching is enabled.'
            )
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.batch_per_gpu = batch_per_gpu
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.generator = generator
        self.curriculum = curriculum
        self.epoch = 1
        if curriculum is not None:
            self.curriculum_start_max = float(curriculum['start_max'])
            self.curriculum_end_epoch = int(curriculum['end_epoch'])
            if not self.scale_min <= self.curriculum_start_max <= self.scale_max:
                raise ValueError(
                    'curriculum start_max must be within the scale range.'
                )
            if self.curriculum_end_epoch < 1:
                raise ValueError('curriculum end_epoch must be positive.')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def current_scale_max(self):
        if self.curriculum is None:
            return self.scale_max
        if self.curriculum_end_epoch == 1:
            return self.scale_max
        progress = min(
            max((self.epoch - 1) / (self.curriculum_end_epoch - 1), 0.0),
            1.0,
        )
        return (
            self.curriculum_start_max
            + progress * (self.scale_max - self.curriculum_start_max)
        )

    def __len__(self):
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self):
        current_scale_max = self.current_scale_max()
        indices = torch.randperm(
            self.dataset_size, generator=self.generator
        ).tolist()
        for start in range(0, self.dataset_size, self.batch_size):
            batch = indices[start:start + self.batch_size]
            grouped = []
            for group_start in range(0, len(batch), self.batch_per_gpu):
                scale = torch.empty(1).uniform_(
                    self.scale_min,
                    current_scale_max,
                    generator=self.generator,
                ).item()
                grouped.extend(
                    (index, scale)
                    for index in batch[group_start:group_start + self.batch_per_gpu]
                )
            yield grouped


def capture_random_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'data_generators': {
            name: generator.get_state()
            for name, generator in data_generators.items()
        },
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_random_state(state):
    if state is None:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state.get('cuda') is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    for name, generator_state in state.get('data_generators', {}).items():
        if name in data_generators:
            data_generators[name].set_state(generator_state)


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        if not isinstance(v, float):
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    num_workers = spec.get('num_workers', config.get('num_workers', 8))
    loader_generator = make_generator(
        '{}_loader'.format(tag), training_seed, 1000 if tag == 'train' else 2000
    )
    loader_args = {
        'num_workers': num_workers,
        'pin_memory': True,
        'worker_init_fn': seed_worker,
        'generator': loader_generator,
    }
    if tag == 'train' and spec.get('scale_grouped_batching', False):
        wrapper_args = spec['wrapper']['args']
        sampler_generator = make_generator(
            'train_sampler', training_seed, 3000
        )
        loader_args['batch_sampler'] = ScaleGroupedBatchSampler(
            dataset_size=len(dataset),
            batch_size=spec['batch_size'],
            batch_per_gpu=wrapper_args['batch_per_gpu'],
            scale_min=wrapper_args['scale_min'],
            scale_max=wrapper_args['scale_max'],
            generator=sampler_generator,
            curriculum=spec.get('scale_curriculum'),
        )
        loader = DataLoader(dataset, **loader_args)
    else:
        loader = DataLoader(
            dataset,
            batch_size=spec['batch_size'],
            shuffle=(tag == 'train'),
            **loader_args
        )
    return loader, dataset


def make_data_loaders():
    train_loader, train_dataset = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader, val_dataset = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader, train_dataset, val_dataset


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        checkpoint_seed = sv_file.get('seed')
        if checkpoint_seed is not None and checkpoint_seed != training_seed:
            raise ValueError(
                'Resume checkpoint seed {} does not match requested seed {}.'
                .format(checkpoint_seed, training_seed)
            )
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)  #
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        if lr_scheduler is not None:
            if sv_file.get('lr_scheduler') is not None:
                lr_scheduler.load_state_dict(sv_file['lr_scheduler'])
            else:
                for _ in range(epoch_start - 1):
                    lr_scheduler.step()
        random_state = sv_file.get('random_state')
        max_val_v = sv_file.get('best_val', -1e18)
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])  #
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        random_state = None
        max_val_v = -1e18

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler, random_state, max_val_v


def train(train_loader, model, optimizer):
    model.train()
    train_loss = utils.Averager()

    data_norm = config['data_norm']
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        pred = model(inp, batch['coord'], batch['scale'], batch['cell'])

        gt = (batch['gt'] - gt_sub) / gt_div
        pixel_loss = (pred - gt).abs()
        if 'weight' in batch:
            loss = (pixel_loss * batch['weight']).mean()
        else:
            loss = pixel_loss.mean()
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None
        loss = None

    return train_loss.item()


def main(config_, save_path, seed):
    global config, log, writer, training_seed
    config = copy.deepcopy(config_)
    training_seed = seed
    config['seed'] = seed
    deterministic = bool(config.get('deterministic', True))
    set_seed(seed, deterministic=deterministic)
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader, train_dataset, val_dataset = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler, random_state, max_val_v = prepare_training()
    if random_state is None:
        # Extra modules consume different amounts of RNG during construction.
        # Reset here so shared training-time randomness is aligned across A/B.
        set_seed(seed, deterministic=deterministic)
    else:
        restore_random_state(random_state)
    log('seed={}, deterministic={}'.format(seed, deterministic))
    if config.get('train_dataset', {}).get('scale_grouped_batching', False):
        log('scale-grouped batching: enabled')

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
            if train_loader.batch_sampler.curriculum is not None:
                log_info.append(
                    'scale_max={:.2f}'.format(
                        train_loader.batch_sampler.current_scale_max()
                    )
                )

        train_loss = train(train_loader, model, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        is_best = False
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                                data_norm=config['data_norm'],
                                eval_type=config.get('eval_type'),
                                eval_bsize=config.get('eval_bsize'))

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            if val_res > max_val_v:
                max_val_v = val_res
                is_best = True

        model_spec = copy.deepcopy(config['model'])
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = copy.deepcopy(config['optimizer'])
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch,
            'seed': training_seed,
            'best_val': max_val_v,
            'random_state': capture_random_state(),
            'lr_scheduler': (
                None if lr_scheduler is None else lr_scheduler.state_dict()
            ),
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))
        if is_best:
            torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    seed = args.seed if args.seed is not None else config.get('seed')
    if seed is None:
        raise ValueError(
            'A training seed is required. Set seed in YAML or pass --seed.'
        )

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, seed)
