import argparse
import os
import sys

import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
OFFICIAL = os.path.join(ROOT, 'third_party', 'gsasr_paper_9d2eb64')
sys.path.insert(0, OFFICIAL)

from utils.edsrbaseline import EDSRNOUP  # noqa: E402
from utils.fea2gs import Fea2GS, grid_to_tokens, tokens_to_grid  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    encoder_state = torch.load(
        os.path.join(args.weights, 'encoder.pth'), map_location='cpu'
    )['params_ema']
    decoder_state = torch.load(
        os.path.join(args.weights, 'decoder.pth'), map_location='cpu'
    )['params_ema']
    encoder = EDSRNOUP()
    decoder = Fea2GS()
    encoder.load_state_dict(encoder_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    encoder_parameters = sum(value.numel() for value in encoder.parameters())
    decoder_parameters = sum(value.numel() for value in decoder.parameters())
    if encoder_parameters != 1220416 or decoder_parameters != 19224381:
        raise RuntimeError('Official model parameter count mismatch.')

    tokens = torch.arange(2 * 2 * 3 * 4 * 4 * 5).reshape(
        2 * 2 * 3, 4 * 4, 5
    )
    grid = tokens_to_grid(tokens, 2, 3, 4)
    recovered = grid_to_tokens(grid, 2, 3, 4)
    if not torch.equal(tokens, recovered):
        raise RuntimeError('Token/grid compatibility transform is invalid.')
    print('GSASR OFFICIAL SUBSET TEST PASSED')
    print('parameters: encoder={}, decoder={}'.format(
        encoder_parameters, decoder_parameters
    ))


if __name__ == '__main__':
    main()
