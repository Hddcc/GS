#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from common.face_metrics import metrics, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--first-k", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import datasets
    from test import load_model_from_config, sample_chop

    config = OmegaConf.load(str(args.config))
    config.model.params.ignore_keys = ["loss"]
    config.model.params.ckpt_path = str(args.checkpoint.resolve())
    model = load_model_from_config(config, str(args.checkpoint)).eval()
    images = sorted(args.images.glob("*.png"))
    if args.first_k:
        images = images[:args.first_k]
    if not images:
        raise FileNotFoundError("No PNG test images in " + str(args.images))
    data = datasets.make({"name": "image-folder", "args": {
        "root_path": str(args.images.resolve()), "first_k": len(images)}})
    rows = []
    for scale in args.scales:
        wrapper = datasets.make({"name": "sr-implicit-downsampled", "args": {
            "inp_size": None, "scale_min": scale, "scale_max": scale, "phase": "valid"}},
            args={"dataset": data})
        for index, path in enumerate(images):
            batch = wrapper.collate_fn([wrapper[index]])
            batch = {key: value.cuda() for key, value in batch.items()}
            lr, gt = batch["inp"], batch["gt"]
            coord, cell = batch["coord"], batch["cell"]
            with torch.no_grad():
                cond = model.get_learned_conditioning(lr)
                with model.ema_scope():
                    samples = sample_chop(model, cond, 1, args.steps, eta=0.0)
                pred = model.decode_first_stage(
                    samples, lr, coord=coord, cell=cell, fid_feat=cond,
                    bsize=30000, return_img=False)
                pred = pred.view(1, coord.shape[1], coord.shape[2], 3).permute(0, 3, 1, 2)
                pred = (pred[..., :gt.shape[-2], :gt.shape[-1]] * 0.5 + 0.5).clamp(0, 1)
                gt = gt * 0.5 + 0.5
            psnr, ssim = metrics(pred, gt, scale)
            rows.append({"image": path.name, "scale": scale, "crop_border": math.ceil(scale),
                         "psnr_y": psnr, "ssim_y": ssim})
            if (index + 1) % 100 == 0:
                print("x{:g}: {}/{}".format(scale, index + 1, len(images)), flush=True)
    save_results(rows, args.output)


if __name__ == "__main__":
    main()
