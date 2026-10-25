#!/usr/bin/env python3
"""Compare FTD-LIIF coordinate paths on one checkpoint and image."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

from common.face_metrics import metrics
from evaluate_ftd import coord_axis
from native_ftd_check import EXPECTED_CHECKPOINT_SHA256, make_dataset


ROOT = Path(__file__).resolve().parent


def evaluation_coords(height, width, device):
    axis_x = coord_axis(height, device)
    axis_y = coord_axis(width, device)
    grid_x, grid_y = torch.meshgrid(axis_x, axis_y, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    cell_x = torch.full_like(axis_x, 2 / height)
    cell_y = torch.full_like(axis_y, 2 / width)
    return grid, axis_x.unsqueeze(0), axis_y.unsqueeze(0), cell_x.unsqueeze(0), cell_y.unsqueeze(0)


def native_coords(sample, device):
    return tuple(sample[key].unsqueeze(0).to(device) for key in
                 ("coord", "coordx", "coordy", "cellx", "celly"))


def max_difference(left, right):
    return max((a - b).abs().max().item() for a, b in zip(left, right))


def read_reference(path, image, scale):
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["image"] == image and float(row["scale"]) == scale:
                return {"psnr_y": float(row["psnr_y"]), "ssim_y": float(row["ssim_y"])}
    return None


def diagnose(model, inp, gt, native, evaluated, scale, gt_sub, gt_div):
    cases = {
        "native": native,
        "evaluated": evaluated,
        "native_grid_evaluated_axes": (native[0], evaluated[1], evaluated[2], native[3], native[4]),
        "evaluated_grid_native_axes": (evaluated[0], native[1], native[2], native[3], native[4]),
    }
    predictions = {}
    scores = {}
    with torch.inference_mode():
        for name, coords in cases.items():
            pred = model(inp, *coords)
            pred = (pred * gt_div + gt_sub).clamp(0, 1)
            predictions[name] = pred
            psnr, ssim = metrics(pred, gt, scale)
            scores[name] = {"psnr_y": psnr, "ssim_y": ssim}
        repeat = model(inp, *native)
        repeat = (repeat * gt_div + gt_sub).clamp(0, 1)
    scores["native_repeat_max_pixel_delta"] = (repeat - predictions["native"]).abs().max().item()
    scores["native_evaluated_max_pixel_delta"] = (
        predictions["native"] - predictions["evaluated"]
    ).abs().max().item()
    scores["native_evaluated_mean_pixel_delta"] = (
        predictions["native"] - predictions["evaluated"]
    ).abs().mean().item()
    scores["coordinate_max_delta"] = max_difference(native, evaluated)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "ftd_liif/source")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "ftd_liif/source/save/face_ftd_liif/epoch-last.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "ftd_liif/config/face_ftd_liif.yaml")
    parser.add_argument("--images", type=Path, default=ROOT.parent / "datasets/face/CelebA/test/HR")
    parser.add_argument("--image", default="201418.png")
    parser.add_argument("--scales", type=float, nargs="+", default=[3.5, 4.0])
    parser.add_argument("--reference-per-image", type=Path,
                        default=ROOT / "ftd_liif/results/diagnostic_scales_20_per_image.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ftd_liif/results/inference_diagnostic_20261025.json")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("Refusing to overwrite: " + str(args.output))
    for path in (args.checkpoint, args.config, args.images / args.image):
        if not path.is_file():
            raise FileNotFoundError(path)
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Checkpoint SHA256 mismatch: " + digest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the saved FTD-LIIF checkpoint")

    sys.path.insert(0, str(args.source.resolve()))
    import models

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["epoch"] != 1000:
        raise ValueError("Expected epoch 1000, found {}".format(checkpoint["epoch"]))
    model = models.make(checkpoint["model"], load_sd=True).cuda().eval()
    norm = yaml.safe_load(args.config.read_text(encoding="utf-8"))["data_norm"]
    inp_sub = torch.tensor(norm["inp"]["sub"], device="cuda").view(1, -1, 1, 1)
    inp_div = torch.tensor(norm["inp"]["div"], device="cuda").view(1, -1, 1, 1)
    gt_sub = torch.tensor(norm["gt"]["sub"], device="cuda").view(1, -1, 1, 1)
    gt_div = torch.tensor(norm["gt"]["div"], device="cuda").view(1, -1, 1, 1)

    results = {"checkpoint_sha256": digest, "checkpoint_epoch": checkpoint["epoch"],
               "image": args.image, "scales": {}}
    for scale in args.scales:
        dataset, names = make_dataset(args.source, args.images, scale, 20)
        if args.image not in names:
            raise ValueError("Image not in first 20 sorted test images: " + args.image)
        sample = dataset[names.index(args.image)]
        inp = (sample["inp"].unsqueeze(0).cuda() - inp_sub) / inp_div
        gt = sample["gt"].unsqueeze(0).cuda()
        native = native_coords(sample, inp.device)
        height, width = gt.shape[-2:]
        evaluated = evaluation_coords(height, width, inp.device)
        score = diagnose(model, inp, gt, native, evaluated, scale, gt_sub, gt_div)
        score["reference"] = read_reference(args.reference_per_image, args.image, scale)
        results["scales"][str(scale)] = score
        print("x{:g}: {}".format(scale, json.dumps(score, sort_keys=True)), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("diagnostic=" + str(args.output), flush=True)


if __name__ == "__main__":
    main()
