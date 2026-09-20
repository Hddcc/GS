from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-geometry-guided-appearance')
class FaceGeometryGuidedAppearanceGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """GaussianSR with geometry-conditioned appearance restoration."""
    pass
