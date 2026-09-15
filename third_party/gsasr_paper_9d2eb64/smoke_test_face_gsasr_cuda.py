import argparse
import os

import torch

from evaluate_face_gsasr_candidate import infer, load_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    device = torch.device('cuda')
    encoder, decoder = load_models(args.weights, device)
    torch.manual_seed(1)
    lr = torch.rand(1, 3, 12, 12, device=device)
    with torch.no_grad():
        first = infer(encoder, decoder, lr, 2, 24, 24)
        second = infer(encoder, decoder, lr, 2, 24, 24)
    if first.shape != (1, 3, 24, 24) or not torch.isfinite(first).all():
        raise RuntimeError('GSASR CUDA smoke output is invalid.')
    repeat_error = (first - second).abs().max().item()
    if repeat_error > 1e-6:
        raise RuntimeError('GSASR CUDA smoke is not deterministic.')
    print('GSASR CUDA SMOKE TEST PASSED')
    print('output: {}'.format(tuple(first.shape)))
    print('repeat max error: {:.8e}'.format(repeat_error))


if __name__ == '__main__':
    main()
