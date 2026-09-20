import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from datasets.wrappers import resize_fn


torch.backends.cudnn.enabled = False
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:2048')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batched_predict(model, inp, coord, scale, cell, bsize):
    model.gen_feat(inp)
    preds = []
    for left in range(0, coord.shape[1], bsize):
        right = min(left + bsize, coord.shape[1])
        preds.append(model.query_rgb(
            coord[:, left:right, :].contiguous(),
            scale.contiguous(),
            cell[:, left:right, :].contiguous(),
        ))
    return torch.cat(preds, dim=1)


def load_data_norm(model_path, config_path=None):
    if config_path is None:
        candidate = Path(model_path).resolve().parent / 'config.yaml'
        config_path = candidate if candidate.exists() else None

    if config_path is None:
        return {
            'inp': {'sub': [0.5], 'div': [0.5]},
            'gt': {'sub': [0.5], 'div': [0.5]},
        }

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config.get('data_norm', {
        'inp': {'sub': [0.5], 'div': [0.5]},
        'gt': {'sub': [0.5], 'div': [0.5]},
    })


def make_loader(root_path, scale, num_workers, first_k=None):
    dataset = datasets.make({
        'name': 'image-folder',
        'args': {
            'root_path': root_path,
            'cache': 'none',
            'first_k': first_k,
        },
    })
    filenames = [Path(x).name for x in dataset.files]
    dataset = datasets.make({
        'name': 'sr-implicit-downsampled',
        'args': {
            'inp_size': None,
            'scale_min': scale,
            'scale_max': scale,
            'augment': False,
            'sample_q': None,
            'batch_per_gpu': 1,
        },
    }, args={'dataset': dataset})
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, filenames


def reshape_samples(samples, inp, scale):
    height = round(inp.shape[-2] * scale)
    width = round(inp.shape[-1] * scale)
    expected = height * width
    if samples.shape[1] != expected:
        raise ValueError(
            'Cannot reshape {} samples to {}x{} at scale {}.'.format(
                samples.shape[1], height, width, scale
            )
        )
    return samples.view(samples.shape[0], height, width, 3) \
        .permute(0, 3, 1, 2).contiguous()


def crop_border(image, border):
    if border == 0:
        return image
    if image.shape[-2] <= 2 * border or image.shape[-1] <= 2 * border:
        raise ValueError(
            'Crop border {} is too large for image shape {}.'.format(
                border, tuple(image.shape)
            )
        )
    return image[..., border:-border, border:-border]


def rgb_to_y(image):
    coeffs = image.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    return image.mul(coeffs).sum(dim=1, keepdim=True) + 16 / 255


def calculate_psnr_y(sr, gt, border):
    # Offset is omitted from the difference, matching utils.calc_psnr benchmark mode.
    coeffs = sr.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    diff = (sr - gt).mul(coeffs).sum(dim=1, keepdim=True)
    diff = crop_border(diff, border)
    mse = diff.pow(2).mean().item()
    if mse == 0:
        return float('inf')
    return -10 * math.log10(mse)


def calculate_ssim_y(sr, gt, border):
    sr_y = crop_border(rgb_to_y(sr), border)[0, 0].detach().cpu().numpy()
    gt_y = crop_border(rgb_to_y(gt), border)[0, 0].detach().cpu().numpy()
    return float(structural_similarity(
        gt_y,
        sr_y,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    ))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_scale(args, model, lpips_model, data_norm, scale):
    loader, filenames = make_loader(
        args.dataset_root, scale, args.num_workers, args.first_k
    )

    inp_cfg = data_norm['inp']
    gt_cfg = data_norm['gt']
    inp_sub = torch.tensor(inp_cfg['sub'], dtype=torch.float32, device='cuda').view(1, -1, 1, 1)
    inp_div = torch.tensor(inp_cfg['div'], dtype=torch.float32, device='cuda').view(1, -1, 1, 1)
    gt_sub = torch.tensor(gt_cfg['sub'], dtype=torch.float32, device='cuda').view(1, 1, -1)
    gt_div = torch.tensor(gt_cfg['div'], dtype=torch.float32, device='cuda').view(1, 1, -1)

    border = math.ceil(scale)
    rows = []
    progress = tqdm(loader, desc='{} x{}'.format(args.dataset_name, scale), leave=False)
    for index, batch in enumerate(progress):
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        actual_scale = float(batch['scale'][0])
        with torch.no_grad():
            gt = reshape_samples(batch['gt'], batch['inp'], actual_scale)
            if args.method == 'model':
                inp = (batch['inp'] - inp_sub) / inp_div
                pred = batched_predict(
                    model,
                    inp,
                    batch['coord'],
                    batch['scale'],
                    batch['cell'],
                    args.eval_bsize,
                )
                pred = (pred * gt_div + gt_sub).clamp_(0, 1)
                pred = reshape_samples(pred, batch['inp'], actual_scale)
            else:
                pred = resize_fn(
                    batch['inp'][0].cpu(), gt.shape[-2:]
                ).unsqueeze(0).cuda()

            psnr_y = calculate_psnr_y(pred, gt, border)
            ssim_y = calculate_ssim_y(pred, gt, border)
            if lpips_model is None:
                lpips_value = float('nan')
            else:
                lpips_value = float(lpips_model(pred * 2 - 1, gt * 2 - 1).mean().item())

        row = {
            'method': args.method,
            'dataset': args.dataset_name,
            'image': filenames[index],
            'scale': scale,
            'crop_border': border,
            'psnr_y': psnr_y,
            'ssim_y': ssim_y,
            'lpips': lpips_value,
        }
        rows.append(row)
        progress.set_postfix(
            psnr='{:.4f}'.format(np.mean([x['psnr_y'] for x in rows])),
            ssim='{:.4f}'.format(np.mean([x['ssim_y'] for x in rows])),
            lpips='n/a' if lpips_model is None else '{:.4f}'.format(
                np.mean([x['lpips'] for x in rows])
            ),
        )

    summary = {
        'method': args.method,
        'dataset': args.dataset_name,
        'scale': scale,
        'images': len(rows),
        'crop_border': border,
        'psnr_y': float(np.mean([x['psnr_y'] for x in rows])),
        'ssim_y': float(np.mean([x['ssim_y'] for x in rows])),
        'lpips': float(np.mean([x['lpips'] for x in rows])) if lpips_model is not None else float('nan'),
    }
    return summary, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate face SR with Y-PSNR, Y-SSIM, and RGB LPIPS.'
    )
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--dataset-name', required=True)
    parser.add_argument('--method', choices=['model', 'bicubic'], default='model')
    parser.add_argument('--model', default=None)
    parser.add_argument('--scales', type=float, nargs='+', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=None)
    parser.add_argument('--eval-bsize', type=int, default=10000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--first-k', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lpips-net', choices=['alex', 'vgg', 'squeeze'], default='alex')
    parser.add_argument('--skip-lpips', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_seed(args.seed)

    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError('Dataset directory not found: {}'.format(args.dataset_root))
    if args.method == 'model' and not args.model:
        raise ValueError('--model is required when --method=model.')
    if args.method == 'model' and not os.path.isfile(args.model):
        raise FileNotFoundError('Checkpoint not found: {}'.format(args.model))
    if any(scale < 1 for scale in args.scales):
        raise ValueError('All scales must be at least 1.')

    if args.method == 'model':
        checkpoint = torch.load(args.model, map_location='cpu')
        model = models.make(checkpoint['model'], load_sd=True).cuda().eval()
        data_norm = load_data_norm(args.model, args.config)
    else:
        model = None
        data_norm = load_data_norm('', args.config)

    if args.skip_lpips:
        lpips_model = None
    else:
        import lpips
        lpips_model = lpips.LPIPS(net=args.lpips_net).cuda().eval()

    summaries = []
    per_image_rows = []
    for scale in args.scales:
        summary, rows = evaluate_scale(
            args, model, lpips_model, data_norm, scale
        )
        summaries.append(summary)
        per_image_rows.extend(rows)
        print(
            '{} x{:g}: PSNR-Y={:.4f}, SSIM-Y={:.6f}, LPIPS={}'.format(
                args.dataset_name,
                scale,
                summary['psnr_y'],
                summary['ssim_y'],
                'n/a' if args.skip_lpips else '{:.6f}'.format(summary['lpips']),
            )
        )

    write_csv(
        args.output,
        ['method', 'dataset', 'scale', 'images', 'crop_border', 'psnr_y', 'ssim_y', 'lpips'],
        summaries,
    )
    output_path = Path(args.output)
    per_image_path = output_path.with_name(output_path.stem + '_per_image' + output_path.suffix)
    write_csv(
        per_image_path,
        ['method', 'dataset', 'image', 'scale', 'crop_border', 'psnr_y', 'ssim_y', 'lpips'],
        per_image_rows,
    )
    print('Summary CSV: {}'.format(output_path))
    print('Per-image CSV: {}'.format(per_image_path))


if __name__ == '__main__':
    main()
