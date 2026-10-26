#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from common.face_metrics import metrics, save_results


def image_tensor(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def coord_axis(size, device):
    radius = 1 / size
    axis = -1 + radius + (2 * radius) * torch.arange(size, dtype=torch.float32)
    return axis.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--first-k", type=int)
    args = parser.parse_args()
    per_image = args.output.with_name(args.output.stem + "_per_image" + args.output.suffix)
    for path in (args.output, per_image):
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing result: " + str(path))
    sys.path.insert(0, str(args.source.resolve()))
    import models

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    norm = config["data_norm"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = models.make(checkpoint["model"], load_sd=True).to(device).eval()
    files = sorted(args.images.glob("*.png"))
    if args.first_k:
        files = files[:args.first_k]
    if not files:
        raise FileNotFoundError("No PNG test images in " + str(args.images))
    rows = []
    for scale in args.scales:
        for index, path in enumerate(files, 1):
            with Image.open(path) as image:
                width, height = image.size
                low_h = math.floor(height / scale + 1e-9)
                low_w = math.floor(width / scale + 1e-9)
                out_h = round(low_h * scale)
                out_w = round(low_w * scale)
                image = image.crop((0, 0, out_w, out_h))
                gt = image_tensor(image).unsqueeze(0).to(device)
                low = image.resize((low_w, low_h), Image.Resampling.BICUBIC)
                inp = image_tensor(low).unsqueeze(0).to(device)
            sub = torch.tensor(norm["inp"]["sub"], device=device).view(1, -1, 1, 1)
            div = torch.tensor(norm["inp"]["div"], device=device).view(1, -1, 1, 1)
            inp = (inp - sub) / div
            coordx = coord_axis(out_h, inp.device)
            coordy = coord_axis(out_w, inp.device)
            cy, cx = torch.meshgrid(coordx, coordy, indexing="ij")
            coord = torch.stack((cy, cx), dim=-1).unsqueeze(0)
            cellx = torch.full_like(coordx, 2 / out_h)
            celly = torch.full_like(coordy, 2 / out_w)
            with torch.no_grad():
                pred = model(inp, coord, coordx.unsqueeze(0), coordy.unsqueeze(0),
                             cellx.unsqueeze(0), celly.unsqueeze(0))
                pred = (pred * norm["gt"]["div"][0] + norm["gt"]["sub"][0]).clamp(0, 1)
            psnr, ssim = metrics(pred, gt, scale)
            rows.append({"image": path.name, "scale": scale, "crop_border": math.ceil(scale),
                         "psnr_y": psnr, "ssim_y": ssim})
            if index % 100 == 0:
                print("x{:g}: {}/{}".format(scale, index, len(files)), flush=True)
    save_results(rows, args.output)


if __name__ == "__main__":
    main()
