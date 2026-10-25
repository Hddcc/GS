#!/usr/bin/env python3
"""Cross-check FTD-LIIF inference with its upstream dataset wrapper."""

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import torch
import yaml

from common.face_metrics import metrics, save_results


ROOT = Path(__file__).resolve().parent
EXPECTED_CHECKPOINT_SHA256 = "60d4b210206cd521d5d875df66745d754a208a5743bfa8e696e4967676e0b2f4"


def make_dataset(source, images, scale, first_k):
    sys.path.insert(0, str(source.resolve()))
    from datasets.image_folder import ImageFolder
    from datasets.wrappers import SRImplicitDownsampledLRTFR

    files = sorted(images.glob("*.png"))[:first_k]
    if len(files) != first_k:
        raise ValueError("Expected {} PNG images, found {}".format(first_k, len(files)))
    dataset = ImageFolder(str(images), first_k=first_k)
    names = [Path(path).name for path in dataset.files]
    if names != [path.name for path in files]:
        raise ValueError("Native image order differs from the reference evaluation")
    return SRImplicitDownsampledLRTFR(
        dataset, inp_size=None, scale_min=scale, scale_max=scale,
        augment=False, sample_q=None,
    ), names


def compare_rows(rows, reference_path, psnr_tolerance=0.05, ssim_tolerance=0.002):
    with reference_path.open(newline="", encoding="utf-8") as handle:
        reference = {
            (row["image"], float(row["scale"])): row
            for row in csv.DictReader(handle)
        }
    observed = {(row["image"], row["scale"]): row for row in rows}
    if observed.keys() != reference.keys():
        raise ValueError("Native and reference image/scale pairs differ")

    max_psnr = 0.0
    max_ssim = 0.0
    failures = []
    for key, row in observed.items():
        expected = reference[key]
        psnr_delta = abs(row["psnr_y"] - float(expected["psnr_y"]))
        ssim_delta = abs(row["ssim_y"] - float(expected["ssim_y"]))
        max_psnr = max(max_psnr, psnr_delta)
        max_ssim = max(max_ssim, ssim_delta)
        if psnr_delta > psnr_tolerance or ssim_delta > ssim_tolerance:
            failures.append((key, psnr_delta, ssim_delta))

    print("native-reference max deltas: PSNR-Y={:.6f}, SSIM-Y={:.6f}".format(
        max_psnr, max_ssim), flush=True)
    if failures:
        print("first mismatch: {}".format(failures[0]), flush=True)
        raise ValueError("Native inference does not match the existing evaluation")
    print("NATIVE FTD CROSS-CHECK PASSED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "ftd_liif/source")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "ftd_liif/source/save/face_ftd_liif/epoch-last.pth")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "ftd_liif/config/face_ftd_liif.yaml")
    parser.add_argument("--images", type=Path,
                        default=ROOT.parent / "datasets/face/CelebA/test/HR")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ftd_liif/results/native_check_20_20261025.csv")
    parser.add_argument("--reference-per-image", type=Path,
                        default=ROOT / "ftd_liif/results/diagnostic_scales_20_per_image.csv")
    parser.add_argument("--scales", type=float, nargs="+", default=[3, 3.25, 3.5, 3.75, 4])
    parser.add_argument("--first-k", type=int, default=20)
    parser.add_argument("--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256)
    args = parser.parse_args()

    per_image = args.output.with_name(args.output.stem + "_per_image" + args.output.suffix)
    for path in (args.output, per_image):
        if path.exists():
            raise FileExistsError("Refusing to overwrite existing result: " + str(path))
    for path in (args.checkpoint, args.config, args.reference_per_image):
        if not path.is_file():
            raise FileNotFoundError(path)
    actual_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if actual_hash != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 mismatch: " + actual_hash)
    if not torch.cuda.is_available():
        raise RuntimeError("FTD-LIIF native inference requires a CUDA GPU")

    sys.path.insert(0, str(args.source.resolve()))
    import models

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    print("checkpoint epoch: {}".format(checkpoint["epoch"]), flush=True)
    if checkpoint["epoch"] != 1000:
        raise ValueError("Expected the completed 1000-epoch checkpoint")
    model = models.make(checkpoint["model"], load_sd=True).cuda().eval()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    norm = config["data_norm"]
    inp_sub = torch.tensor(norm["inp"]["sub"], device="cuda").view(1, -1, 1, 1)
    inp_div = torch.tensor(norm["inp"]["div"], device="cuda").view(1, -1, 1, 1)
    gt_sub = torch.tensor(norm["gt"]["sub"], device="cuda").view(1, -1, 1, 1)
    gt_div = torch.tensor(norm["gt"]["div"], device="cuda").view(1, -1, 1, 1)

    rows = []
    for scale in args.scales:
        dataset, names = make_dataset(args.source, args.images, scale, args.first_k)
        for index, name in enumerate(names):
            sample = dataset[index]
            inp = (sample["inp"].unsqueeze(0).cuda() - inp_sub) / inp_div
            coord = sample["coord"].unsqueeze(0).cuda()
            coordx = sample["coordx"].unsqueeze(0).cuda()
            coordy = sample["coordy"].unsqueeze(0).cuda()
            cellx = sample["cellx"].unsqueeze(0).cuda()
            celly = sample["celly"].unsqueeze(0).cuda()
            with torch.no_grad():
                pred = model(inp, coord, coordx, coordy, cellx, celly)
                pred = (pred * gt_div + gt_sub).clamp(0, 1)
            gt = sample["gt"].unsqueeze(0).cuda()
            psnr, ssim = metrics(pred, gt, scale)
            rows.append({"image": name, "scale": scale, "crop_border": math.ceil(scale),
                         "psnr_y": psnr, "ssim_y": ssim})
        print("native x{:g}: {}/{}".format(scale, len(names), len(names)), flush=True)

    save_results(rows, args.output)
    compare_rows(rows, args.reference_per_image)


if __name__ == "__main__":
    main()
