import hashlib
import math
import os

import torch
import torch.nn.functional as F


def covariance_from_parameters(sigma_major, sigma_minor, angle_degrees,
                               device=None, dtype=torch.float32):
    if sigma_major <= 0 or sigma_minor <= 0:
        raise ValueError('Gaussian sigmas must be positive.')
    if sigma_major < sigma_minor:
        raise ValueError('sigma_major must be at least sigma_minor.')
    angle = math.radians(float(angle_degrees))
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], device=device, dtype=dtype)
    diagonal = torch.diag(torch.tensor(
        [sigma_major ** 2, sigma_minor ** 2],
        device=device,
        dtype=dtype,
    ))
    return rotation.matmul(diagonal).matmul(rotation.transpose(0, 1))


def gaussian_kernel(covariance, kernel_size=21):
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('Gaussian kernel size must be a positive odd number.')
    covariance = torch.as_tensor(covariance)
    if covariance.shape != (2, 2):
        raise ValueError('Covariance must have shape [2, 2].')
    radius = kernel_size // 2
    axis = torch.arange(
        -radius, radius + 1,
        device=covariance.device,
        dtype=covariance.dtype,
    )
    yy, xx = torch.meshgrid(axis, axis)
    xy = torch.stack([xx, yy], dim=-1)
    inverse = torch.linalg.inv(covariance)
    exponent = torch.einsum(
        '...i,ij,...j->...', xy, -0.5 * inverse, xy
    )
    kernel = torch.exp(exponent)
    return kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).tiny)


def kernel_covariance(kernel, coordinate_step=1.0):
    kernel = torch.as_tensor(kernel)
    if kernel.ndim != 2 or min(kernel.shape) <= 0:
        raise ValueError('Kernel must be a non-empty 2D tensor.')
    weights = kernel.clamp_min(0)
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    y_axis = (torch.arange(kernel.shape[0], device=kernel.device,
                           dtype=kernel.dtype) - (kernel.shape[0] - 1) / 2)
    x_axis = (torch.arange(kernel.shape[1], device=kernel.device,
                           dtype=kernel.dtype) - (kernel.shape[1] - 1) / 2)
    yy, xx = torch.meshgrid(y_axis, x_axis)
    coordinates = torch.stack([xx, yy], dim=-1) * float(coordinate_step)
    mean = (weights.unsqueeze(-1) * coordinates).sum(dim=(0, 1))
    centered = coordinates - mean
    return torch.einsum('hw,hwi,hwj->ij', weights, centered, centered)


def apply_anisotropic_blur(image, covariance, kernel_size=21):
    if image.ndim != 3:
        raise ValueError('Image must have shape [C, H, W].')
    kernel = gaussian_kernel(
        torch.as_tensor(covariance, device=image.device, dtype=image.dtype),
        kernel_size,
    )
    radius = kernel_size // 2
    if min(image.shape[-2:]) <= radius:
        raise ValueError('Image is too small for reflected blur padding.')
    weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(
        image.shape[0], 1, 1, 1
    )
    padded = F.pad(image.unsqueeze(0), (radius,) * 4, mode='reflect')
    return F.conv2d(padded, weight, groups=image.shape[0]).squeeze(0)


def lr_effective_covariance(hr_kernel, scale):
    if scale <= 1:
        raise ValueError('Scale must be greater than one.')
    output_size = max(3, round(hr_kernel.shape[-1] / float(scale)))
    if output_size % 2 == 0:
        output_size += 1
    resized = F.interpolate(
        hr_kernel.view(1, 1, *hr_kernel.shape),
        size=(output_size, output_size),
        mode='bicubic',
        align_corners=False,
    ).view(output_size, output_size).clamp_min(0)
    resized = resized / resized.sum().clamp_min(torch.finfo(resized.dtype).tiny)
    return kernel_covariance(resized)


def renderer_kernel_moment(covariance, kernel_size=5):
    covariance = torch.as_tensor(covariance)
    axis = torch.linspace(
        -5.0, 5.0, steps=kernel_size,
        device=covariance.device, dtype=covariance.dtype,
    )
    yy, xx = torch.meshgrid(axis, axis)
    xy = torch.stack([xx, yy], dim=-1)
    inverse = torch.linalg.inv(covariance)
    exponent = torch.einsum(
        '...i,ij,...j->...', xy, -0.5 * inverse, xy
    )
    response = torch.exp(exponent)
    return kernel_covariance(response)


def calibrate_renderer_units(kernel_size=5, reference_variance=2.25,
                             epsilon=0.05):
    if reference_variance <= epsilon or epsilon <= 0:
        raise ValueError('Invalid renderer calibration parameters.')
    dtype = torch.float64
    identity = torch.eye(2, dtype=dtype)
    lower = renderer_kernel_moment(
        identity * (reference_variance - epsilon), kernel_size
    )
    upper = renderer_kernel_moment(
        identity * (reference_variance + epsilon), kernel_size
    )
    pixel_variance_gain = (
        torch.trace(upper - lower) / (4 * epsilon)
    ).item()
    if not math.isfinite(pixel_variance_gain) or pixel_variance_gain <= 0:
        raise RuntimeError('Renderer impulse calibration is not invertible.')
    return 1.0 / pixel_variance_gain


def mapped_internal_covariance(hr_kernel, scale, mapping,
                               renderer_units_per_pixel_variance):
    if mapping == 'hr':
        pixel_covariance = kernel_covariance(hr_kernel)
    elif mapping == 'lr':
        pixel_covariance = lr_effective_covariance(hr_kernel, scale)
        pixel_covariance = pixel_covariance * float(scale) ** 2
    else:
        raise ValueError('Unknown covariance mapping: {}'.format(mapping))
    return pixel_covariance * float(renderer_units_per_pixel_variance)


def oracle_variants(alphas=(0.25, 0.5, 1.0)):
    variants = []
    for mapping in ('hr', 'lr'):
        for mode in ('add', 'subtract'):
            for alpha in alphas:
                variants.append({
                    'name': '{}-{}-a{:g}'.format(mapping, mode, alpha),
                    'mapping': mapping,
                    'mode': mode,
                    'alpha': float(alpha),
                })
    return variants


def expand_manifest(protocol, split_name):
    split = protocol['splits'][split_name]
    degradations = protocol['degradations']
    entries = []
    for order in range(split['count']):
        degradation = degradations[order % len(degradations)]
        entries.append({
            'order': order,
            'dataset_index': split['start_index'] + order,
            'bucket': degradation['bucket'],
            'sigma_major': float(degradation['sigma_major']),
            'sigma_minor': float(degradation['sigma_minor']),
            'angle_degrees': float(degradation['angle_degrees']),
        })
    return entries


def sha256_file(path):
    digest = hashlib.sha256()
    with open(os.fspath(path), 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
