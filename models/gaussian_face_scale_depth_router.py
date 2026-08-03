import torch

from models import register
from models.gaussian import make_coord
from models.gaussian_face_frequency_residual import (
    FaceFrequencyResidualGaussianSplatter,
)


@register('gaussian-splatter-face-scale-depth-router')
class FaceScaleDepthRouterGaussianSplatter(
        FaceFrequencyResidualGaussianSplatter):
    """Pass target scale into spatial cross-depth appearance routing."""

    def _encode_scale_conditioned_features(self, inp, scale):
        self.inp = inp
        self.feat, self.logits = self.encoder(inp, scale)
        self.feat_coord = make_coord(
            inp.shape[-2:], flatten=False
        ).to(inp.device).permute(2, 0, 1).unsqueeze(0).expand(
            inp.shape[0], 2, *inp.shape[-2:]
        )
        self.residual_feature = self.frequency_encoder(self.feat, inp)
        self._encoded_scale = scale.detach().clone()
        return self.feat, self.logits

    def gen_feat(self, inp, scale=None):
        self._pending_inp = inp
        if scale is None:
            self.feat = None
            self.logits = None
            self.feat_coord = None
            self.residual_feature = None
            self._encoded_scale = None
            return None
        return self._encode_scale_conditioned_features(inp, scale)

    def query_rgb(self, coord, scale, cell=None):
        encoded_scale = getattr(self, '_encoded_scale', None)
        scale_changed = (
            encoded_scale is None
            or encoded_scale.shape != scale.shape
            or not torch.equal(encoded_scale, scale)
        )
        if self.feat is None or scale_changed:
            pending_inp = getattr(self, '_pending_inp', None)
            if pending_inp is None:
                raise RuntimeError('gen_feat must be called before query_rgb.')
            self._encode_scale_conditioned_features(pending_inp, scale)
        return super().query_rgb(coord, scale, cell)

    def forward(self, inp, coord, scale, cell=None):
        self.gen_feat(inp, scale)
        return self.query_rgb(coord, scale, cell)
