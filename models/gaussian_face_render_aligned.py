from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-render-aligned')
class FaceRenderAlignedGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """GaussianSR with renderer-aligned feature responsibilities."""
    pass
