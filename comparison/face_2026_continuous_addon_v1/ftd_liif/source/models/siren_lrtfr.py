import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import math
# import matplotlib.pyplot as plt 
import time
from utils import make_coord
from models import register
dtype = torch.cuda.FloatTensor

class SineLayer(nn.Module):
    # This code is from https://github.com/YisiLuo/Continuous-Tensor-Toolbox/blob/main/Demo_PoinCloud_upsampling.py
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=2.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

@register('siren_lrtfr_tucker')
class Siren_LRTFR_Tucker(nn.Module):
    def __init__(self, rank=100, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_Tucker, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN2 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        centre = nn.Parameter(torch.Tensor(3,rank,rank,rank).type(dtype))
        stdv = 1 / math.sqrt(centre.size(1))
        centre.data.uniform_(-stdv, stdv)
        self.centre = centre
        self.centre.requires_grad=True

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)
        
        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r
       
        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w
        
        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('cijk,bni->cbnjk', self.centre, x_factor)
        centre = torch.einsum('cbnjk,bmj->cbnmk', centre, y_factor)
        centre = torch.einsum('cbnmk,bknm->cbknm', centre, cond_factor)
        centre = torch.sum(centre, dim=2)
        centre = centre.permute(1,0,2,3)

        return centre

@register('siren_lrtfr_tucker_l4')
class Siren_LRTFR_Tucker_l4(nn.Module):
    def __init__(self, rank=100, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_Tucker_l4, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN2 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        centre = nn.Parameter(torch.Tensor(3,rank,rank,rank).type(dtype))
        stdv = 1 / math.sqrt(centre.size(1))
        centre.data.uniform_(-stdv, stdv)
        self.centre = centre
        self.centre.requires_grad=True

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)

        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r
        
        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w
        
        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('cijk,bni->cbnjk', self.centre, x_factor)
        centre = torch.einsum('cbnjk,bmj->cbnmk', centre, y_factor)
        centre = torch.einsum('cbnmk,bknm->cbknm', centre, cond_factor)
        centre = torch.sum(centre, dim=2)
        centre = centre.permute(1,0,2,3)

        return centre

@register('siren_lrtfr_tucker_l0')
class Siren_LRTFR_Tucker_l0(nn.Module):
    def __init__(self, rank=100, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_Tucker_l0, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(SineLayer(2, rank, omega_0=omega_0))
        self.TF_SIREN2 = nn.Sequential(SineLayer(2, rank, omega_0=omega_0))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        centre = nn.Parameter(torch.Tensor(3,rank,rank,rank).type(dtype))
        stdv = 1 / math.sqrt(centre.size(1))
        centre.data.uniform_(-stdv, stdv)
        self.centre = centre
        self.centre.requires_grad=True

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)

        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r

        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w

        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('cijk,bni->cbnjk', self.centre, x_factor)
        centre = torch.einsum('cbnjk,bmj->cbnmk', centre, y_factor)
        centre = torch.einsum('cbnmk,bknm->cbknm', centre, cond_factor)
        centre = torch.sum(centre, dim=2)
        centre = centre.permute(1,0,2,3)

        return centre

@register('siren_lrtfr_cp')
class Siren_LRTFR_CP(nn.Module):
    def __init__(self, rank=150, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_CP, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN2 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        self.proj = nn.Linear(rank, 3)

        self.time1 = 0
        self.time2 = 0
        self.time3 = 0

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)

        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r

        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w

        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('bni,bmi->bnmi', x_factor, y_factor)
        centre = torch.einsum('bnmk,bknm->bnmk', centre, cond_factor)
        centre = self.proj(centre).permute(0,3,1,2)

        return centre

@register('siren_lrtfr_cp_l4')
class Siren_LRTFR_CP_l4(nn.Module):
    def __init__(self, rank=100, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_CP_l4, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN2 = nn.Sequential(SineLayer(2, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    SineLayer(300, 300, omega_0=omega_0),
                                    nn.Linear(300, rank))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        self.proj = nn.Linear(rank, 3)

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)

        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r
        
        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w

        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('bni,bmi->bnmi', x_factor, y_factor)
        centre = torch.einsum('bnmk,bknm->bnmk', centre, cond_factor)
        centre = self.proj(centre).permute(0,3,1,2)

        return centre
    
@register('siren_lrtfr_cp_l0')
class Siren_LRTFR_CP_l0(nn.Module):
    def __init__(self, rank=100, omega_0=100, up_mode='bilinear'):
        super(Siren_LRTFR_CP_l0, self).__init__()
        self.partial_functions = nn.ModuleList()
        self.up_mode = up_mode

        self.TF_SIREN1 = nn.Sequential(
                                    nn.Linear(2, rank))
        self.TF_SIREN2 = nn.Sequential(
                                    nn.Linear(2, rank))
        self.TF_SIREN3 = nn.Sequential(nn.Linear(576, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, 376),
                                       nn.ReLU(),
                                       nn.Linear(376, rank))
        
        self.proj = nn.Linear(rank, 3)

    def forward(self, inputs, feat, return_stats=False):
        coordx, coordy, coord = inputs[0], inputs[1], inputs[2] # (B N c) (B N N 2)
        
        cond_input = self.TF_SIREN3(
            feat.permute(0,2,3,1)) # B, h, w, r

        cond_input = cond_input.permute(0,3,1,2) # B,r,h,w
        
        cond_factor = F.grid_sample(
            cond_input, coord.flip(-1),
            mode=self.up_mode, align_corners=False).permute(0,1,2,3) # B,r,h,w
        t2 = time.time()

        x_factor = self.TF_SIREN1(coordx) # B,N,r
        y_factor = self.TF_SIREN2(coordy) # B,N,r

        centre = torch.einsum('bni,bmi->bnmi', x_factor, y_factor)
        centre = torch.einsum('bnmk,bknm->bnmk', centre, cond_factor)
        centre = self.proj(centre).permute(0,3,1,2)

        return centre
