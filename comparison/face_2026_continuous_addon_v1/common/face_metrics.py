import csv
import math
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity


def metrics(pred, target, scale):
    border = math.ceil(scale)
    pred = pred[..., border:-border, border:-border]
    target = target[..., border:-border, border:-border]
    coeffs = pred.new_tensor([65.738, 129.057, 25.064]).view(1, 3, 1, 1) / 256
    pred_y = (pred * coeffs).sum(dim=1, keepdim=True) + 16 / 255
    target_y = (target * coeffs).sum(dim=1, keepdim=True) + 16 / 255
    mse = (pred_y - target_y).square().mean().item()
    psnr = float("inf") if mse == 0 else -10 * math.log10(mse)
    ssim = structural_similarity(
        target_y[0, 0].detach().cpu().numpy(),
        pred_y[0, 0].detach().cpu().numpy(),
        data_range=1.0, gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False,
    )
    return psnr, float(ssim)


def save_results(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scales = sorted({row["scale"] for row in rows})
    summary = []
    for scale in scales:
        part = [row for row in rows if row["scale"] == scale]
        result = {
            "scale": scale, "images": len(part), "crop_border": math.ceil(scale),
            "psnr_y": float(np.mean([row["psnr_y"] for row in part])),
            "ssim_y": float(np.mean([row["ssim_y"] for row in part])),
        }
        summary.append(result)
        print("x{:g}: PSNR-Y={:.6f}, SSIM-Y={:.6f}".format(
            scale, result["psnr_y"], result["ssim_y"]), flush=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    per_image = output.with_name(output.stem + "_per_image" + output.suffix)
    with per_image.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("summary=" + str(output), flush=True)
    print("per_image=" + str(per_image), flush=True)
