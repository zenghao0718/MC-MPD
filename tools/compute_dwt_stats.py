"""Compute train-set DWT high-frequency mean/std for the dual-branch experiment."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.frequency import haar_dwt_highfreq_rgb_mean


DEFAULT_FOLDERS = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute DWT high-frequency stats from training split only.")
    parser.add_argument("--data_root", type=str, required=True, help="Root containing real/ ADM/ BigGAN/ ... folders.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS, help="Dataset folders to include.")
    parser.add_argument("--exclude_class", type=str, default=None, help="Optional folder to exclude, matching training.")
    parser.add_argument("--resize_size", type=int, default=256, help="Deterministic Resize size before CenterCrop.")
    parser.add_argument("--crop_size", type=int, default=224, help="CenterCrop size.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples_per_folder", type=int, default=-1,
                        help="Use all samples when negative; otherwise sample up to this many per folder.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dwt_input_scale", type=float, default=1.0)
    parser.add_argument("--dwt_use_abs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dwt_use_log1p", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_dataset(folder_path, transform, max_samples, seed):
    dataset = ImageFolder(folder_path, transform=transform)
    if max_samples is None or max_samples < 0 or max_samples >= len(dataset):
        return dataset, len(dataset)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[:max_samples]
    return Subset(dataset, indices), len(indices)


def main():
    args = parse_args()
    transform = transforms.Compose(
        [
            transforms.Resize(args.resize_size),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
        ]
    )

    folders = [folder for folder in args.folders if folder != args.exclude_class]
    total_sum = torch.zeros(3, dtype=torch.float64)
    total_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0
    total_images = 0
    per_folder_counts = {}

    with torch.no_grad():
        for folder in folders:
            train_path = os.path.join(args.data_root, folder, "train")
            if not os.path.isdir(train_path):
                raise FileNotFoundError(f"Training folder not found: {train_path}")

            dataset, used_count = build_dataset(
                train_path,
                transform,
                args.max_samples_per_folder,
                args.seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=False,
                drop_last=False,
            )

            per_folder_counts[folder] = used_count
            for images, _ in tqdm(loader, desc=f"DWT stats: {folder}"):
                freq = haar_dwt_highfreq_rgb_mean(
                    images,
                    use_abs=args.dwt_use_abs,
                    use_log1p=args.dwt_use_log1p,
                    resize_to=None,
                    mean=None,
                    std=None,
                    input_scale=args.dwt_input_scale,
                ).double()
                total_sum += freq.sum(dim=(0, 2, 3))
                total_sum_sq += (freq ** 2).sum(dim=(0, 2, 3))
                total_pixels += freq.shape[0] * freq.shape[2] * freq.shape[3]
                total_images += freq.shape[0]

    if total_pixels == 0:
        raise RuntimeError("No training images were found for DWT stats.")

    mean = total_sum / total_pixels
    var = (total_sum_sq / total_pixels) - mean ** 2
    std = var.clamp_min(0).sqrt()

    output_data = {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "num_images": int(total_images),
        "per_folder_counts": per_folder_counts,
        "preprocess": {
            "resize_size": args.resize_size,
            "crop_size": args.crop_size,
            "steps": "Resize -> CenterCrop -> ToTensor -> DWT -> abs -> RGB mean over subbands -> log1p",
        },
        "dwt_input_scale": args.dwt_input_scale,
        "dwt_use_abs": args.dwt_use_abs,
        "dwt_use_log1p": args.dwt_use_log1p,
        "split": "train only",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        f.write("\n")

    print(f"Saved DWT stats to {output_path}")
    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    main()
