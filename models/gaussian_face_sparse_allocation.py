from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-sparse-allocation')
class FaceSparseAllocationGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """GaussianSR with support-adaptive sparse parameter allocation."""
    pass
