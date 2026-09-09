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


def finite_mean(name, values):
    value = statistics.mean(values)
    if not math.isfinite(value):
        raise RuntimeError('{} is not finite.'.format(name))
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
    if getattr(candidate, 'primitive_count', None) != 4:
        raise RuntimeError('Candidate checkpoint does not use m=4.')
    if candidate.gaussian_channels != 8:
        raise RuntimeError('Candidate checkpoint changed Gaussian channels.')
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    candidate_parameters = sum(p.numel() for p in candidate.parameters())
    if candidate_parameters - baseline_parameters != 2100:
        raise RuntimeError('Candidate has the wrong parameter count.')

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
    branch_responses = [[] for _ in range(4)]
    offset_displacements = []
    center_separations = []
    covariance_stds = []
    opacity_minimums = []
    opacity_stds = []
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

                diagnostic = candidate.last_primitive_diagnostics
                if diagnostic is None:
                    raise RuntimeError('Primitive diagnostics are missing.')
                for branch, value in enumerate(diagnostic['branch_response']):
                    branch_responses[branch].append(value.item())
                offsets = diagnostic['offsets']
                offset_displacements.append(
                    (offsets - candidate.base_offsets).abs().mean().item()
                )
                distances = torch.pdist(offsets)
                center_separations.append(distances.min().item())
                covariance_stds.append(
                    diagnostic['covariance_scales'].std().item()
                )
                opacity = diagnostic['opacity_weights']
                opacity_minimums.append(opacity.min().item())
                opacity_stds.append(opacity.std().item())
                output_responses.append(
                    diagnostic['unclamped_output_mean_abs'].item()
                )

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
        baseline_parameters,
        candidate_parameters,
        candidate_parameters - baseline_parameters,
    ))
    branch_means = []
    for branch, values in enumerate(branch_responses, 1):
        value = finite_mean('branch {} response'.format(branch), values)
        branch_means.append(value)
        print('primitive {} color response mean abs: {:.8f}'.format(branch, value))
    diversity = statistics.stdev(branch_means) / max(
        statistics.mean(branch_means), 1e-12
    )
    print('normalized primitive diversity: {:.8f}'.format(diversity))
    print('offset displacement mean abs: {:.8f}'.format(
        finite_mean('offset displacement', offset_displacements)
    ))
    print('minimum primitive center distance: {:.8f}'.format(
        min(center_separations)
    ))
    print('covariance scale spatial std: {:.8f}'.format(
        finite_mean('covariance scale std', covariance_stds)
    ))
    print('minimum opacity weight: {:.8f}'.format(min(opacity_minimums)))
    print('opacity weight spatial std: {:.8f}'.format(
        finite_mean('opacity std', opacity_stds)
    ))
    print('unclamped renderer response mean abs: {:.8f}'.format(
        finite_mean('renderer response', output_responses)
    ))
    print('MULTI-PRIMITIVE EVALUATION COMPLETED')


if __name__ == '__main__':
    main()
