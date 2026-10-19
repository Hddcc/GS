#!/usr/bin/env python3
"""Generate CelebA configs from pinned upstream configs."""

import argparse
from pathlib import Path

import yaml


def read_yaml(path):
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def prepare_ftd(args):
    root = args.addon / "ftd_liif"
    source = root / "source" / "configs" / "train-liiflrtfr"
    config = read_yaml(source / "train_edsr-baseline-liiflrtfr-cp.yaml")
    train = config["train_dataset"]
    val = config["val_dataset"]
    train["dataset"]["args"].update(root_path=str(args.train), repeat=1)
    val["dataset"]["args"].update(root_path=str(args.val), first_k=50, repeat=1)
    for section, low, high in ((train, 2.0, 8.0), (val, 4.0, 4.0)):
        section["wrapper"]["args"].update(
            inp_size=16, sample_q=32, scale_min=low, scale_max=high
        )
    train["batch_size"] = args.batch_size
    val["batch_size"] = min(args.batch_size, 4)
    config["model"]["args"]["encoder_spec"] = {
        "name": "edsr-baseline", "args": {"no_upsampling": True}
    }
    config["epoch_max"] = args.epochs
    config["epoch_val"] = 1
    config["epoch_save"] = 50
    if args.resume:
        config["resume"] = str(args.resume)
    config["multi_step_lr"]["milestones"] = [
        max(1, round(args.epochs * fraction)) for fraction in (0.2, 0.4, 0.6, 0.8)
    ]
    write_yaml(root / "config" / "face_ftd_liif.yaml", config)


def prepare_fplia(args):
    root = args.addon / "fplia"
    upstream = root / "source" / "configs"
    out = root / "config"
    stages = ("lia-swinir", "vae", "diffusion", "finetune")
    previous = {
        "vae": args.lia_checkpoint,
        "diffusion": args.vae_checkpoint,
        "finetune": args.diffusion_checkpoint,
    }
    for stage in stages:
        if args.stage and stage != args.stage:
            continue
        config = read_yaml(upstream / (stage + ".yaml"))
        data = config["data"]["params"]
        data["batch_size"] = args.batch_size if stage == "lia-swinir" else 1
        data["num_workers"] = 2
        for section, path in (("train", args.train), ("validation", args.val)):
            spec = data[section]
            dataset = spec["dataset"]["params"]
            dataset["root_path"] = str(path)
            dataset["repeat"] = 1
            if section == "validation":
                dataset["first_k"] = 50
            else:
                dataset.pop("first_k", None)
            wrapper = spec["wrapper"]["params"]
            wrapper["inp_size"] = 16
            wrapper["scale_min"] = [4.0]
            wrapper["scale_max"] = [4.0]
            if section == "train" and stage in ("lia-swinir", "finetune"):
                wrapper["scale_min"] = [2.0, 4.0]
                wrapper["scale_max"] = [4.0, 8.0]
                wrapper["k"] = 2
            if "sample_q" in wrapper:
                wrapper["sample_q"] = 1024
        model = config["model"]["params"]
        if "sample_q" in model:
            model["sample_q"] = 1024
        if stage == "vae":
            model["fidconfig"] = {"name": "swinir", "args": {"no_upsampling": True}}
        if stage in ("diffusion", "finetune"):
            model["image_size"] = [16, 16]
        if stage in previous:
            if not previous[stage]:
                raise ValueError("Missing prerequisite checkpoint for " + stage)
            if stage == "diffusion":
                model["first_stage_config"]["params"]["ckpt_path"] = str(previous[stage])
            else:
                model["ckpt_path"] = str(previous[stage])
        config["lightning"]["trainer"].update(
            devices="0,", accelerator="gpu", precision=32,
            max_epochs=args.epochs, accumulate_grad_batches=1
        )
        write_yaml(out / ("face_" + stage + ".yaml"), config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("ftd", "fplia"))
    parser.add_argument("--stage", choices=("lia-swinir", "vae", "diffusion", "finetune"))
    parser.add_argument("--addon", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lia-checkpoint", type=Path)
    parser.add_argument("--vae-checkpoint", type=Path)
    parser.add_argument("--diffusion-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.addon = args.addon.resolve()
    args.train = args.train.resolve()
    args.val = args.val.resolve()
    for path in (args.train, args.val):
        if not path.is_dir() or not any(path.glob("*.png")):
            raise FileNotFoundError("Missing PNG face dataset: " + str(path))
    if args.method == "ftd":
        prepare_ftd(args)
    else:
        prepare_fplia(args)


if __name__ == "__main__":
    main()
