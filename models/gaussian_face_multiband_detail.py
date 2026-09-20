from models import register
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-multiband-detail')
class FaceMultiBandDetailGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """GaussianSR with explicit multi-band appearance detail features."""
    pass
