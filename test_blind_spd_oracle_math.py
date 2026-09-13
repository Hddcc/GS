import math

import torch

from blind_spd_oracle import (
    apply_anisotropic_blur,
    calibrate_renderer_units,
    covariance_from_parameters,
    gaussian_kernel,
    kernel_covariance,
    lr_effective_covariance,
    mapped_internal_covariance,
    oracle_variants,
)


def main():
    covariance = covariance_from_parameters(1.6, 0.8, 45.0)
    eigenvalues = torch.linalg.eigvalsh(covariance)
    if not torch.allclose(eigenvalues, torch.tensor([0.64, 2.56]), atol=1e-5):
        raise RuntimeError('Anisotropic covariance construction failed.')
    kernel = gaussian_kernel(covariance, 21)
    if abs(kernel.sum().item() - 1) > 1e-6 or kernel.min().item() < 0:
        raise RuntimeError('Gaussian kernel normalization failed.')
    measured = kernel_covariance(kernel)
    if (measured - covariance).abs().max().item() > 2e-4:
        raise RuntimeError('Discrete Gaussian covariance is inaccurate.')

    constant = torch.full((3, 31, 33), 0.37)
    blurred = apply_anisotropic_blur(constant, covariance, 21)
    if (blurred - constant).abs().max().item() > 2e-6:
        raise RuntimeError('Blur does not preserve a constant image.')

    lr_covariance = lr_effective_covariance(kernel, 4.0)
    if not torch.isfinite(lr_covariance).all():
        raise RuntimeError('LR effective covariance is not finite.')
    if torch.linalg.eigvalsh(lr_covariance).min().item() < -1e-7:
        raise RuntimeError('LR effective covariance is not PSD.')

    calibration = calibrate_renderer_units(5, 2.25, 0.05)
    if not math.isfinite(calibration) or calibration <= 0:
        raise RuntimeError('Renderer impulse calibration failed.')
    for mapping in ('hr', 'lr'):
        mapped = mapped_internal_covariance(kernel, 4.0, mapping, calibration)
        if not torch.isfinite(mapped).all():
            raise RuntimeError('{} mapping is not finite.'.format(mapping))
        if torch.linalg.eigvalsh(mapped).min().item() < -1e-7:
            raise RuntimeError('{} mapping is not PSD.'.format(mapping))

    variants = oracle_variants((0.25, 0.5, 1.0))
    if len(variants) != 12 or len({item['name'] for item in variants}) != 12:
        raise RuntimeError('Oracle variant grid is incomplete.')
    print('renderer units per pixel variance: {:.8f}'.format(calibration))
    print('BLIND-SPD ORACLE MATH TEST PASSED')


if __name__ == '__main__':
    main()
