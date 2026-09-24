import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def parse_model_specs(values):
    specs = []
    for value in values:
        if '=' not in value:
            raise ValueError('Each --models value must use label=checkpoint.')
        label, checkpoint = value.split('=', 1)
        label = label.strip()
        checkpoint = checkpoint.strip()
        if not label or not checkpoint:
            raise ValueError('Each --models value must use label=checkpoint.')
        specs.append((label, checkpoint))
    return specs


def save_tensor_image(tensor, path):
    image = tensor.detach().cpu().clamp(0, 1)[0]
    image = image.permute(1, 2, 0).numpy()
    image = np.rint(image * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Export identical face SR visual comparisons for several checkpoints.'
    )
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--models', nargs='+', required=True,
                        help='label=checkpoint pairs, for example baseline=save/.../epoch-best.pth')
    parser.add_argument('--scales', type=float, nargs='+', required=True)
    parser.add_argument('--indices', type=int, nargs='+', required=True,
                        help='Zero-based test image indices.')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--eval-bsize', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    import datasets
    import models
    from datasets.wrappers import resize_fn
    from test_face_metrics import batched_predict, load_data_norm, make_loader, reshape_samples

    if min(args.indices) < 0:
        raise ValueError('--indices must be non-negative.')
    model_specs = parse_model_specs(args.models)
    max_index = max(args.indices)
    selected = set(args.indices)
    loaded_models = []
    for label, checkpoint_path in model_specs:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError('Checkpoint not found: {}'.format(checkpoint))
        state = torch.load(checkpoint, map_location='cpu')
        network = models.make(state['model'], load_sd=True).cuda().eval()
        loaded_models.append({
            'label': label,
            'model': network,
            'data_norm': load_data_norm(str(checkpoint)),
        })

    output_root = Path(args.output_dir)
    for scale in args.scales:
        loader, filenames = make_loader(
            args.dataset_root, scale, args.num_workers, first_k=max_index + 1
        )
        inp_cfg = loaded_models[0]['data_norm']['inp']
        gt_cfg = loaded_models[0]['data_norm']['gt']
        inp_sub = torch.tensor(inp_cfg['sub'], dtype=torch.float32, device='cuda').view(1, -1, 1, 1)
        inp_div = torch.tensor(inp_cfg['div'], dtype=torch.float32, device='cuda').view(1, -1, 1, 1)
        gt_sub = torch.tensor(gt_cfg['sub'], dtype=torch.float32, device='cuda').view(1, 1, -1)
        gt_div = torch.tensor(gt_cfg['div'], dtype=torch.float32, device='cuda').view(1, 1, -1)

        for index, batch in enumerate(loader):
            if index > max_index:
                break
            if index not in selected:
                continue
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            actual_scale = float(batch['scale'][0])
            with torch.no_grad():
                gt = reshape_samples(batch['gt'], batch['inp'], actual_scale)
                bicubic = resize_fn(batch['inp'][0].cpu(), gt.shape[-2:]).unsqueeze(0).cuda()
                predictions = {}
                for item in loaded_models:
                    model_norm = item['data_norm']
                    model_inp_sub = torch.tensor(
                        model_norm['inp']['sub'], dtype=torch.float32, device='cuda'
                    ).view(1, -1, 1, 1)
                    model_inp_div = torch.tensor(
                        model_norm['inp']['div'], dtype=torch.float32, device='cuda'
                    ).view(1, -1, 1, 1)
                    model_gt_sub = torch.tensor(
                        model_norm['gt']['sub'], dtype=torch.float32, device='cuda'
                    ).view(1, 1, -1)
                    model_gt_div = torch.tensor(
                        model_norm['gt']['div'], dtype=torch.float32, device='cuda'
                    ).view(1, 1, -1)
                    inp = (batch['inp'] - model_inp_sub) / model_inp_div
                    pred = batched_predict(
                        item['model'], inp, batch['coord'], batch['scale'],
                        batch['cell'], args.eval_bsize
                    )
                    pred = (pred * model_gt_div + model_gt_sub).clamp_(0, 1)
                    predictions[item['label']] = reshape_samples(
                        pred, batch['inp'], actual_scale
                    )

            stem = Path(filenames[index]).stem.replace(' ', '_')
            sample_dir = output_root / ('x{:g}'.format(scale)) / (
                '{:04d}_{}'.format(index, stem)
            )
            save_tensor_image(batch['inp'].detach().cpu(), sample_dir / 'input_lr.png')
            save_tensor_image(bicubic, sample_dir / 'bicubic.png')
            save_tensor_image(gt, sample_dir / 'ground_truth.png')
            for label, _ in model_specs:
                save_tensor_image(predictions[label], sample_dir / '{}.png'.format(label))
            print('{} x{:g} index={} -> {}'.format(
                args.dataset_name, scale, index, sample_dir
            ))


if __name__ == '__main__':
    main()
