import argparse
import hashlib
import json
import math
import os

import torch
import yaml
from PIL import Image
from torchvision import transforms

import datasets
import models
from utils import make_coord


RESIZE_PROTOCOL = 'torchvision-pil-bicubic-uint8-v1'


def make_query(height, width, device):
    coord = make_coord((height, width)).to(device).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / height
    cell[:, :, 1] *= 2 / width
    return coord, cell


def predict(model, inp, coord, scale, cell, eval_bsize):
    model.gen_feat(inp)
    chunks = []
    for start in range(0, coord.shape[1], eval_bsize):
        stop = min(start + eval_bsize, coord.shape[1])
        chunks.append(model.query_rgb(
            coord[:, start:stop].contiguous(),
            scale,
            cell[:, start:stop].contiguous(),
        ))
    return torch.cat(chunks, dim=1)


def psnr(prediction, target):
    mse = (prediction - target).square().mean().item()
    return -10 * math.log10(max(mse, 1e-12))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def resize_with_protocol(image, size):
    resized = transforms.Resize(size, Image.BICUBIC)(
        transforms.ToPILImage()(image)
    )
    return transforms.ToTensor()(resized), bytes_sha256(resized.tobytes())


def atomic_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--eval-bsize', type=int, default=10000)
    args = parser.parse_args()
    if args.max_samples <= 0 or args.eval_bsize <= 0:
        raise ValueError('Evaluation sizes must be positive.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if checkpoint.get('seed') != 1:
        raise RuntimeError('Baseline checkpoint must use seed 1.')
    device = torch.device('cuda')
    model = models.make(checkpoint['model'], load_sd=True).to(device).eval()
    if hasattr(model, 'set_oracle_covariance_adjustment'):
        model.set_oracle_covariance_adjustment(None)
    dataset = datasets.make(config['val_dataset']['dataset'])
    if len(dataset) < args.max_samples:
        raise RuntimeError('Validation set contains too few samples.')
    root = config['val_dataset']['dataset']['args']['root_path']
    filenames = sorted(os.listdir(root))[:args.max_samples]

    inp_norm = config['data_norm']['inp']
    gt_norm = config['data_norm']['gt']
    inp_sub = torch.tensor(inp_norm['sub'], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor(inp_norm['div'], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor(gt_norm['sub'], device=device).view(1, 1, -1)
    gt_div = torch.tensor(gt_norm['div'], device=device).view(1, 1, -1)
    scales = (2, 4, 8)
    records = []
    with torch.no_grad():
        for index in range(args.max_samples):
            image = dataset[index]
            height = image.shape[-2] // 8 * 8
            width = image.shape[-1] // 8 * 8
            target_image = image[:, :height, :width].contiguous()
            target = target_image.view(3, -1).permute(1, 0).unsqueeze(0).to(device)
            coord, cell = make_query(height, width, device)
            source_path = os.path.join(root, filenames[index])
            for scale_value in scales:
                lr, lr_sha256 = resize_with_protocol(
                    target_image,
                    (round(height / scale_value), round(width / scale_value)),
                )
                lr_height, lr_width = lr.shape[-2:]
                lr = lr.unsqueeze(0).to(device)
                inp = (lr - inp_sub) / inp_div
                scale = torch.tensor([float(scale_value)], device=device)
                output = predict(
                    model, inp, coord, scale, cell, args.eval_bsize
                )
                output = (output * gt_div + gt_sub).clamp(0, 1)
                value = psnr(output, target)
                records.append({
                    'index': index,
                    'filename': filenames[index],
                    'source_sha256': file_sha256(source_path),
                    'height': height,
                    'width': width,
                    'scale': scale_value,
                    'lr_height': lr_height,
                    'lr_width': lr_width,
                    'lr_sha256': lr_sha256,
                    'psnr': value,
                })
            if (index + 1) % 10 == 0 or index + 1 == args.max_samples:
                print('BASELINE progress: {}/{} images'.format(
                    index + 1, args.max_samples), flush=True)
    result = {
        'protocol': 'face-gsasr-zero-train-v1',
        'model': 'GaussianSR face baseline seed 1 epoch 5',
        'samples': args.max_samples,
        'scales': list(scales),
        'resize_protocol': RESIZE_PROTOCOL,
        'checkpoint_sha256': file_sha256(args.checkpoint),
        'records': records,
    }
    atomic_json(args.output, result)
    for scale_value in scales:
        values = [r['psnr'] for r in records if r['scale'] == scale_value]
        print('BASELINE x{} PSNR: {:.8f}'.format(
            scale_value, sum(values) / len(values)))
    print('GSASR ZERO-TRAIN BASELINE EVALUATION COMPLETED')


if __name__ == '__main__':
    main()
