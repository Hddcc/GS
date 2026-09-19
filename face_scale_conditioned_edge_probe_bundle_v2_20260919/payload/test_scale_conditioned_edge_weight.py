import torch

from datasets.wrappers import (
    effective_edge_weight_strength,
    make_edge_weight_map,
)


def main():
    fixed = effective_edge_weight_strength(0.25, 2.0)
    assert abs(fixed - 0.25) < 1e-12

    low = effective_edge_weight_strength(
        0.25, 2.0, mode='scale-conditioned', scale_reference=4.0,
        scale_power=0.5)
    high = effective_edge_weight_strength(
        0.25, 8.0, mode='scale-conditioned', scale_reference=4.0,
        scale_power=0.5)
    assert low < 0.25 < high

    image = torch.zeros(3, 8, 8)
    image[:, :, 4:] = 1.0
    low_map = make_edge_weight_map(image, low)
    high_map = make_edge_weight_map(image, high)
    assert torch.isfinite(low_map).all()
    assert torch.isfinite(high_map).all()
    assert abs(float(low_map.mean()) - 1.0) < 1e-6
    assert abs(float(high_map.mean()) - 1.0) < 1e-6
    assert not torch.allclose(low_map, high_map)

    invalid_calls = (
        lambda: effective_edge_weight_strength(0.25, 4.0, mode='unknown'),
        lambda: effective_edge_weight_strength(
            0.25, 4.0, scale_reference=0),
        lambda: effective_edge_weight_strength(0.25, 0),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid edge-weight settings were accepted')

    print('SCALE-CONDITIONED EDGE WEIGHT TEST PASSED')


if __name__ == '__main__':
    main()
