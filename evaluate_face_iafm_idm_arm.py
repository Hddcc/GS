import argparse
import math
import statistics

import torch
import yaml

import datasets
import models
from datasets.wrappers import resize_fn
from utils import make_coord


def make_query(height, width, device):
    coord = make_coord((height, width)).to(device).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / height
    cell[:, :, 1] *= 2 / width
    return coord, cell


def predict(model, inp, coord, scale, cell, eval_bsize):
    model.gen_feat(inp)
    predictions = []
    for start in range(0, coord.shape[1], eval_bsize):
        stop = min(start + eval_bsize, coord.shape[1])
        predictions.append(model.query_rgb(
            coord[:, start:stop].contiguous(),
            scale,
            cell[:, start:stop].contiguous(),
        ))
    return torch.cat(predictions, dim=1)


def psnr(prediction, target):
    mse = (prediction - target).square().mean().item()
    return -10 * math.log10(max(mse, 1e-12))


def load_model(path, expected_seed, device):
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint.get('seed') != expected_seed:
        raise RuntimeError('{} has the wrong seed.'.format(path))
    return models.make(checkpoint['model'], load_sd=True).to(device).eval()


def finite_mean(label, values):
    value = statistics.mean(values)
    if not math.isfinite(value):
        raise RuntimeError('{} is not finite.'.format(label))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline-checkpoint', required=True)
    parser.add_argument('--candidate-checkpoint', required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--eval-bsize', type=int, default=10000)
    parser.add_argument('--max-samples', type=int, default=100)
    args = parser.parse_args()
    if args.eval_bsize <= 0 or args.max_samples <= 0:
        raise ValueError('Evaluation sizes must be positive.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable.')

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    expected_seed = config['seed']
    device = torch.device(args.device)
    baseline = load_model(args.baseline_checkpoint, expected_seed, device)
    candidate = load_model(args.candidate_checkpoint, expected_seed, device)
    if candidate.encoder.__class__.__name__ != 'IAFMIDMARMEncoder':
        raise RuntimeError('Candidate checkpoint does not use IAFM IDM+ARM.')
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    if candidate_parameters - baseline_parameters != 151040:
        raise RuntimeError('Candidate has the wrong parameter count.')
    for model in (baseline, candidate):
        if hasattr(model, 'set_oracle_covariance_adjustment'):
            model.set_oracle_covariance_adjustment(None)

    dataset = datasets.make(config['val_dataset']['dataset'])
    if len(dataset) < args.max_samples:
        raise RuntimeError('Validation set contains too few samples.')
    inp_norm = config['data_norm']['inp']
    gt_norm = config['data_norm']['gt']
    inp_sub = torch.tensor(inp_norm['sub'], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor(inp_norm['div'], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor(gt_norm['sub'], device=device).view(1, 1, -1)
    gt_div = torch.tensor(gt_norm['div'], device=device).view(1, 1, -1)

    scales = (2.0, 4.0, 8.0)
    baseline_psnr = {scale: [] for scale in scales}
    candidate_psnr = {scale: [] for scale in scales}
    entropy_values = []
    idm_means = []
    idm_spatial_stds = []
    arm_responses = [[] for _ in candidate.encoder.arm_positions]
    scaled_arm_responses = [[] for _ in candidate.encoder.arm_positions]
    output_responses = []
    with torch.no_grad():
        for index in range(args.max_samples):
            image = dataset[index]
            height = image.shape[-2] // 8 * 8
            width = image.shape[-1] // 8 * 8
            if height == 0 or width == 0:
                raise RuntimeError('Validation image is smaller than 8 pixels.')
            target_image = image[:, :height, :width].contiguous()
            target = target_image.view(3, -1).permute(1, 0).unsqueeze(0).to(device)
            coord, cell = make_query(height, width, device)
            for scale_value in scales:
                lr = resize_fn(
                    target_image,
                    (round(height / scale_value), round(width / scale_value)),
                ).unsqueeze(0).to(device)
                inp = (lr - inp_sub) / inp_div
                scale = torch.tensor([scale_value], device=device)
                baseline_output = predict(
                    baseline, inp, coord, scale, cell, args.eval_bsize
                )
                candidate_output = predict(
                    candidate, inp, coord, scale, cell, args.eval_bsize
                )
                baseline_output = (baseline_output * gt_div + gt_sub).clamp(0, 1)
                candidate_output = (candidate_output * gt_div + gt_sub).clamp(0, 1)
                baseline_psnr[scale_value].append(psnr(baseline_output, target))
                candidate_psnr[scale_value].append(psnr(candidate_output, target))
                output_responses.append(
                    (candidate_output - baseline_output).abs().mean().item()
                )

                diagnostic = candidate.encoder.last_iafm_diagnostics
                if diagnostic is None:
                    raise RuntimeError('IAFM diagnostics are missing.')
                entropy_values.append(
                    candidate.encoder.last_information_entropy_loss.item()
                )
                idm_means.append(diagnostic['idm_mean'].item())
                idm_spatial_stds.append(diagnostic['idm_spatial_std'].item())
                for position, response in enumerate(diagnostic['arm_response']):
                    arm_responses[position].append(response.item())
                for position, response in enumerate(
                        diagnostic['scaled_arm_response']):
                    scaled_arm_responses[position].append(response.item())

    print('seed:', expected_seed)
    print('samples:', args.max_samples)
    print('held-out scales: 2, 4, 8')
    for scale_value in scales:
        baseline_mean = finite_mean(
            'baseline x{:g} PSNR'.format(scale_value),
            baseline_psnr[scale_value],
        )
        candidate_mean = finite_mean(
            'candidate x{:g} PSNR'.format(scale_value),
            candidate_psnr[scale_value],
        )
        print('x{:g} baseline psnr: {:.8f}'.format(scale_value, baseline_mean))
        print('x{:g} candidate psnr: {:.8f}'.format(scale_value, candidate_mean))
        print('x{:g} psnr delta: {:+.8f}'.format(
            scale_value, candidate_mean - baseline_mean
        ))
    print('parameters: baseline={}, candidate={}, delta={}'.format(
        baseline_parameters, candidate_parameters,
        candidate_parameters - baseline_parameters,
    ))
    print('information entropy mean: {:.8f}'.format(
        finite_mean('information entropy', entropy_values)))
    print('IDM mean: {:.8f}'.format(finite_mean('IDM mean', idm_means)))
    print('IDM spatial std: {:.8f}'.format(
        finite_mean('IDM spatial std', idm_spatial_stds)))
    for index, position in enumerate(candidate.encoder.arm_positions):
        print('ARM {} response mean abs: {:.8f}'.format(
            position, finite_mean(
                'ARM {} response'.format(position), arm_responses[index]
            )
        ))
        print('ARM {} scaled response mean abs: {:.8f}'.format(
            position, finite_mean(
                'ARM {} scaled response'.format(position),
                scaled_arm_responses[index],
            )
        ))
    print('candidate-baseline output response mean abs: {:.8f}'.format(
        finite_mean('candidate output response', output_responses)))
    print('IAFM IDM+ARM EVALUATION COMPLETED')


if __name__ == '__main__':
    main()
