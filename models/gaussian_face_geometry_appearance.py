from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-geometry-appearance')
class FaceGeometryAppearanceGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """GaussianSR with decoupled geometry and appearance feature duties."""
    pass
