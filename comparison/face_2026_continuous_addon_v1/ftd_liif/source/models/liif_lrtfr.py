import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord
import time


@register('liif-lrtfr')
class LIIF_LRTFR(nn.Module):
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        if encoder_spec['name'] == 'pretrain':
            encoder_spec = torch.load(encoder_spec['path'])['model']
            self.encoder = models.make(encoder_spec, load_sd=True)
        else:
            self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            self.imnet = models.make(imnet_spec)

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, coordx, coordy, cellx=None, celly=None):
        feat = self.feat

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coordx_ = coordx.clone()
                coordx_ += vx * rx + eps_shift
                coordx_.clamp_(-1 + 1e-6, 1 - 1e-6) # B N

                coordy_ = coordy.clone()
                coordy_ += vy * ry + eps_shift
                coordy_.clamp_(-1 + 1e-6, 1 - 1e-6) # B N

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6) # B N N 2

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1),
                    mode='nearest', align_corners=False
                ) # B 2 N N

                q_coord_x = q_coord[:, 0, :, 0].unsqueeze(-1) # B N 1
                q_coord_y = q_coord[:, 1, 0, :].unsqueeze(-1) # B N 1

                rel_coord = coord - q_coord.permute(0,2,3,1) # B N N 2
                rel_coord_x = coordx_.unsqueeze(-1) - q_coord_x
                rel_coord_y = coordy_.unsqueeze(-1) - q_coord_y
                rel_coord_x *= feat.shape[-2]
                rel_coord_y *= feat.shape[-1]

                inpx, inpy = rel_coord_x, rel_coord_y # B N 1

                if self.cell_decode:

                    rel_cellx, rel_celly = cellx.clone().unsqueeze(-1), celly.clone().unsqueeze(-1) # B N 1
                    rel_cellx *= feat.shape[-2]
                    rel_celly *= feat.shape[-1]

                    inpx = torch.cat([inpx, rel_cellx], dim=-1) # B N 2
                    inpy = torch.cat([inpy, rel_celly], dim=-1) # B N 2

                pred = self.imnet((inpx, inpy, coord), feat)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1]) # B N N
                areas.append(area + 1e-9)
                # areas.append()

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(1) # (B,3,N,N) (B,N,N)
        return ret

    def forward(self, inp, coord, coordx, coordy, cellx, celly):
        self.gen_feat(inp)
        out =  self.query_rgb(coord, coordx, coordy, cellx, celly)
        self.feat = None
        return out
