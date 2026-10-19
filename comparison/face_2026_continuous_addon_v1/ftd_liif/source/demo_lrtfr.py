import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord

import random
import numpy as np
def seed_everything(seed = 1):
    print('seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--input', default='figs/input.png')
    parser.add_argument('--resolution', default='476, 720')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    seed_everything(3407)

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB'))

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))

    coordx = make_coord((h,1), flatten=False)[:,0,0]
    coordy = make_coord((w,1), flatten=False)[:,0,0]
    coord = torch.stack(torch.meshgrid(coordx, coordy), dim=-1)
    cellx = torch.ones_like(coordx)
    celly = torch.ones_like(coordy)
    cellx *= 2/h
    celly *= 2/w

    coordx = coordx.cuda().unsqueeze(0)
    coordy = coordy.cuda().unsqueeze(0)
    cellx = cellx.cuda().unsqueeze(0)
    celly = celly.cuda().unsqueeze(0)
    coord = coord.cuda().unsqueeze(0)

    model.eval()
    pred = model(((img - 0.5) / 0.5).cuda().unsqueeze(0), coord, coordx, coordy, cellx, celly)

    print(pred.shape)
    pred = (pred * 0.5 + 0.5)[0].clamp(0,1).cpu()

    transforms.ToPILImage()(pred).save(args.output)
