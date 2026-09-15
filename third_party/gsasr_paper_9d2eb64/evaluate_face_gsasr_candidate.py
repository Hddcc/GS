import argparse
import hashlib
import json
import math
import os
import time

from PIL import Image
import torch
from torchvision import transforms

from utils.edsrbaseline import EDSRNOUP
from utils.fea2gs import Fea2GS
from utils.gaussian_splatting import generate_2D_gaussian_splatting_step


RESIZE_PROTOCOL = 'torchvision-pil-bicubic-uint8-v1'


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pil_to_tensor(image, device):
    storage = torch.ByteStorage.from_buffer(image.tobytes())
    tensor = torch.ByteTensor(storage).view(image.height, image.width, 3)
    return tensor.permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)


def resize_with_protocol(image, size, device):
    tensor = transforms.ToTensor()(image)
    resized = transforms.Resize(size, Image.BICUBIC)(
        transforms.ToPILImage()(tensor)
    )
    return pil_to_tensor(resized, device), hashlib.sha256(
        resized.tobytes()
    ).hexdigest()


def load_models(weights, device):
    encoder_state = torch.load(
        os.path.join(weights, 'encoder.pth'), map_location='cpu'
    )['params_ema']
    decoder_state = torch.load(
        os.path.join(weights, 'decoder.pth'), map_location='cpu'
    )['params_ema']
    encoder = EDSRNOUP()
    decoder = Fea2GS()
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    return encoder.to(device).eval(), decoder.to(device).eval()


def pad_to_window(inp, denominator=12):
    height, width = inp.shape[-2:]
    pad_h = (denominator - height % denominator) % denominator
    pad_w = (denominator - width % denominator) % denominator
    return torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), 'reflect')


def infer(encoder, decoder, lr, scale, height, width):
    padded = pad_to_window(lr)
    sr_size = torch.tensor([
        math.floor(scale * padded.shape[-2]),
        math.floor(scale * padded.shape[-1]),
    ])
    features = encoder(padded)
    scale_vector = torch.tensor([float(scale)], device=lr.device)
    parameters = decoder(features, scale_vector)[0]
    output = generate_2D_gaussian_splatting_step(
        sr_size=sr_size,
        gs_parameters=parameters,
        scale=float(scale),
        scale_modify=torch.tensor([float(scale), float(scale)]),
        sample_coords=None,
        default_step_size=1.2,
        cuda_rendering=True,
        mode='scale_modify',
        if_dmax=True,
        dmax_mode='fix',
        dmax=0.1,
    )
    return output[:, :height, :width].unsqueeze(0).clamp(0, 1)


def psnr(prediction, target):
    mse = (prediction - target).square().mean().item()
    return -10 * math.log10(max(mse, 1e-12))


def atomic_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-json', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, choices=[2, 4, 8], required=True)
    parser.add_argument('--max-samples', type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    with open(args.baseline_json, 'r', encoding='utf-8') as file:
        baseline = json.load(file)
    if baseline.get('protocol') != 'face-gsasr-zero-train-v1' \
            or baseline.get('resize_protocol') != RESIZE_PROTOCOL:
        raise RuntimeError('Baseline protocol mismatch.')
    reference = [
        record for record in baseline['records']
        if record['scale'] == args.scale
    ]
    if len(reference) != args.max_samples:
        raise RuntimeError('Baseline manifest does not match max-samples.')
    device = torch.device('cuda')
    encoder, decoder = load_models(args.weights, device)
    records = []
    repeat_error = 0.0
    elapsed = 0.0
    with torch.no_grad():
        for index, item in enumerate(reference):
            path = os.path.join(args.data_root, item['filename'])
            if file_sha256(path) != item['source_sha256']:
                raise RuntimeError('Source image hash mismatch: ' + path)
            image = Image.open(path).convert('RGB')
            image = image.crop((0, 0, item['width'], item['height']))
            lr, lr_sha256 = resize_with_protocol(
                image,
                (round(item['height'] / args.scale),
                 round(item['width'] / args.scale)),
                device,
            )
            if lr.shape[-2:] != (item['lr_height'], item['lr_width']) \
                    or lr_sha256 != item['lr_sha256']:
                raise RuntimeError('LR protocol mismatch: ' + path)
            target = pil_to_tensor(image, device)
            torch.cuda.synchronize()
            started = time.time()
            output = infer(
                encoder, decoder, lr, args.scale,
                item['height'], item['width'],
            )
            torch.cuda.synchronize()
            elapsed += time.time() - started
            if index == 0:
                repeated = infer(
                    encoder, decoder, lr, args.scale,
                    item['height'], item['width'],
                )
                repeat_error = (output - repeated).abs().max().item()
            if output.shape != target.shape or not torch.isfinite(output).all():
                raise RuntimeError('Candidate output is invalid.')
            records.append({
                'index': item['index'],
                'filename': item['filename'],
                'source_sha256': item['source_sha256'],
                'height': item['height'],
                'width': item['width'],
                'scale': args.scale,
                'lr_height': item['lr_height'],
                'lr_width': item['lr_width'],
                'lr_sha256': lr_sha256,
                'psnr': psnr(output, target),
            })
            if (index + 1) % 10 == 0 or index + 1 == args.max_samples:
                print('GSASR x{} progress: {}/{} images'.format(
                    args.scale, index + 1, args.max_samples), flush=True)
    result = {
        'protocol': 'face-gsasr-zero-train-v1',
        'model': 'official GSASR paper EDSR DIV2K',
        'official_commit': '9d2eb64a51303ce7a22fd672197185488b38e10a',
        'scale': args.scale,
        'samples': args.max_samples,
        'resize_protocol': RESIZE_PROTOCOL,
        'repeat_max_error': repeat_error,
        'elapsed_seconds': elapsed,
        'encoder_sha256': file_sha256(os.path.join(args.weights, 'encoder.pth')),
        'decoder_sha256': file_sha256(os.path.join(args.weights, 'decoder.pth')),
        'records': records,
    }
    atomic_json(args.output, result)
    mean = sum(record['psnr'] for record in records) / len(records)
    print('GSASR x{} PSNR: {:.8f}'.format(args.scale, mean))
    print('GSASR x{} repeat max error: {:.8e}'.format(
        args.scale, repeat_error))
    print('GSASR x{} elapsed seconds: {:.3f}'.format(args.scale, elapsed))
    print('GSASR x{} CANDIDATE EVALUATION COMPLETED'.format(args.scale))


if __name__ == '__main__':
    main()
