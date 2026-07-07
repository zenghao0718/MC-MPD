"""Extract DDFSD branch features and save t-SNE visualizations."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.amp import autocast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ddfsd_datasets import load_ddfsd_class_dataset, make_subset_loader
from model.ddfsd import DDFSDDualDomainNet


CLASS_ORDER = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
DISPLAY_NAMES = {
    "real": "Real",
    "ADM": "ADM",
    "BigGAN": "BigGAN",
    "glide": "GLIDE",
    "Midjourney": "Midjourney",
    "SD": "Stable Diffusion",
    "VQDM": "VQDM",
}
COLORS = {
    "real": "#D2D0D1",
    "ADM": "#AF7AC5",
    "BigGAN": "#5DADE2",
    "glide": "#F4D03F",
    "Midjourney": "#EC7063",
    "SD": "#58D68D",
    "VQDM": "#F5B041",
}
FEATURE_LABELS = {
    "rgb": "RGB",
    "freq": "Frequency",
    "add": "Add",
    "concat": "Concat",
}
FUSED_NOTE = (
    "add / concat are auxiliary visualization representations only; DDFSD "
    "classification uses branch-wise distance fusion rather than an explicitly "
    "trained fused feature vector."
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="DDFSD t-SNE visualization")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--freq_stats_path", type=str, required=True)
    parser.add_argument("--exclude_class", type=str, required=True, choices=CLASS_ORDER[1:])
    parser.add_argument(
        "--feature_mode",
        type=str,
        required=True,
        choices=["rgb", "freq", "add", "concat"],
    )
    parser.add_argument("--num_samples_each_class", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--use_fp16", type=str2bool, default=True)
    return parser.parse_args()


def ensure_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def validate_args(args) -> None:
    if args.num_samples_each_class <= 0:
        raise ValueError("--num_samples_each_class must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.perplexity <= 0:
        raise ValueError("--perplexity must be positive.")
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"DATA_ROOT does not exist: {args.data_root}")
    ensure_file(args.ckpt_path, "Checkpoint")
    ensure_file(args.freq_stats_path, "Frequency stats")


def sample_index_path(output_dir: Path, exclude_class: str, seed: int) -> Path:
    return output_dir / f"sample_indices_exclude_{exclude_class}_seed{seed}.json"


def load_or_create_sample_indices(
    index_path: Path,
    datasets: Dict[str, object],
    seed: int,
    data_root: str,
    exclude_class: str,
) -> Dict[str, List[int]]:
    """Persist full shuffled class index order so different sample counts can reuse it."""

    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        shuffled_indices = payload.get("shuffled_indices")
        if not isinstance(shuffled_indices, dict):
            raise ValueError(f"Invalid sample index file, missing shuffled_indices: {index_path}")
        for class_name in CLASS_ORDER:
            if class_name not in shuffled_indices:
                raise ValueError(f"Invalid sample index file, missing class: {class_name}")
            if len(shuffled_indices[class_name]) != len(datasets[class_name]):
                raise ValueError(
                    f"Sample index file is incompatible with current dataset for {class_name}: "
                    f"stored {len(shuffled_indices[class_name])} indices, "
                    f"dataset size {len(datasets[class_name])}."
                )
            max_index = max(shuffled_indices[class_name], default=-1)
            if max_index >= len(datasets[class_name]):
                raise ValueError(
                    f"Sample index file is incompatible with current dataset for {class_name}: "
                    f"max index {max_index}, dataset size {len(datasets[class_name])}."
                )
        print(f"Reusing sample index order: {index_path}")
        return {name: list(shuffled_indices[name]) for name in CLASS_ORDER}

    rng = random.Random(seed)
    shuffled_indices = {}
    dataset_sizes = {}
    for class_name in CLASS_ORDER:
        indices = list(range(len(datasets[class_name])))
        rng.shuffle(indices)
        shuffled_indices[class_name] = indices
        dataset_sizes[class_name] = len(indices)

    payload = {
        "seed": seed,
        "exclude_class": exclude_class,
        "data_root": data_root,
        "class_order": CLASS_ORDER,
        "dataset_sizes": dataset_sizes,
        "shuffled_indices": shuffled_indices,
        "note": (
            "Each run takes the first num_samples_each_class indices from this "
            "persisted shuffled order, so all feature modes share identical samples."
        ),
    }
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved sample index order: {index_path}")
    return shuffled_indices


def select_indices(
    shuffled_indices: Dict[str, Sequence[int]],
    datasets: Dict[str, object],
    num_samples_each_class: int,
) -> Dict[str, List[int]]:
    selected = {}
    for class_name in CLASS_ORDER:
        dataset_size = len(datasets[class_name])
        if dataset_size < num_samples_each_class:
            print(
                f"WARNING: {class_name}/val has only {dataset_size} images; "
                f"using all available images."
            )
        count = min(dataset_size, num_samples_each_class)
        selected[class_name] = list(shuffled_indices[class_name][:count])
    return selected


def build_model(args, device: torch.device):
    model = DDFSDDualDomainNet(pretrained=False, freq_stats_path=args.freq_stats_path)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(f"DDFSD checkpoint has no 'model' key: {args.ckpt_path}")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def pick_feature(outputs: Dict[str, torch.Tensor], feature_mode: str) -> torch.Tensor:
    if feature_mode == "rgb":
        return outputs["z_rgb"]
    if feature_mode == "freq":
        return outputs["z_freq"]
    if feature_mode == "add":
        return outputs["z_rgb"] + outputs["z_freq"]
    if feature_mode == "concat":
        return torch.cat([outputs["z_rgb"], outputs["z_freq"]], dim=-1)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


@torch.no_grad()
def extract_features(
    model,
    datasets: Dict[str, object],
    selected_indices: Dict[str, Sequence[int]],
    args,
    device: torch.device,
):
    all_features = []
    all_labels = []
    class_counts = {}
    use_amp = args.use_fp16 and device.type == "cuda"

    for label, class_name in enumerate(CLASS_ORDER):
        loader = make_subset_loader(
            datasets[class_name],
            selected_indices[class_name],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        class_features = []
        for images, _ in loader:
            images = images.to(device=device, non_blocking=True)
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)
                features = pick_feature(outputs, args.feature_mode)
            class_features.append(features.float().detach().cpu())

        if not class_features:
            raise ValueError(f"No features extracted for class: {class_name}")
        features_tensor = torch.cat(class_features, dim=0)
        all_features.append(features_tensor)
        all_labels.append(torch.full((features_tensor.shape[0],), label, dtype=torch.long))
        class_counts[class_name] = int(features_tensor.shape[0])
        print(f"Extracted {features_tensor.shape[0]} samples for {class_name}.")

    features_np = torch.cat(all_features, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    return features_np, labels_np, class_counts


def choose_perplexity(requested: float, num_samples: int) -> float:
    if num_samples <= 1:
        raise ValueError(f"t-SNE needs at least 2 samples, got {num_samples}.")
    max_valid = max(1.0, float(num_samples - 1))
    if requested >= num_samples:
        adjusted = max_valid
        print(
            f"WARNING: requested perplexity {requested} is invalid for {num_samples} "
            f"samples; using {adjusted}."
        )
        return adjusted
    return requested


def run_tsne(features: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        init="pca",
        random_state=seed,
        perplexity=perplexity,
    )
    return tsne.fit_transform(features)


def scatter_tsne(
    tsne_xy: np.ndarray,
    labels: np.ndarray,
    class_counts: Dict[str, int],
    exclude_class: str,
    feature_mode: str,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    for label, class_name in enumerate(CLASS_ORDER):
        mask = labels == label
        legend_name = DISPLAY_NAMES[class_name]
        count = class_counts[class_name]
        if class_name == exclude_class:
            legend_name = f"{legend_name} (test, n={count})"
        else:
            legend_name = f"{legend_name} (n={count})"
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            c=COLORS[class_name],
            s=4,
            label=legend_name,
            alpha=0.82,
            linewidths=0,
        )
    ax.set_title(
        f"DDFSD t-SNE: {FEATURE_LABELS[feature_mode]} | Test on {DISPLAY_NAMES[exclude_class]}"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", title="Classes", fontsize=7, title_fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def metadata_note(feature_mode: str) -> str:
    if feature_mode in {"add", "concat"}:
        return FUSED_NOTE
    return (
        f"{FEATURE_LABELS[feature_mode]} branch feature visualization from DDFSD "
        f"outputs['z_{feature_mode}']."
    )


def main():
    args = parse_args()
    validate_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    seed_output_dir = Path(args.output_dir) / "visualization" / f"seed{args.seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        class_name: load_ddfsd_class_dataset(args.data_root, class_name, "val")
        for class_name in CLASS_ORDER
    }
    index_path = sample_index_path(seed_output_dir, args.exclude_class, args.seed)
    shuffled_indices = load_or_create_sample_indices(
        index_path=index_path,
        datasets=datasets,
        seed=args.seed,
        data_root=args.data_root,
        exclude_class=args.exclude_class,
    )
    selected_indices = select_indices(
        shuffled_indices=shuffled_indices,
        datasets=datasets,
        num_samples_each_class=args.num_samples_each_class,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    features, labels, class_counts = extract_features(
        model=model,
        datasets=datasets,
        selected_indices=selected_indices,
        args=args,
        device=device,
    )
    perplexity_used = choose_perplexity(args.perplexity, features.shape[0])
    tsne_xy = run_tsne(features, args.seed, perplexity_used)

    stem = f"tsne_{args.feature_mode}_exclude_{args.exclude_class}_seed{args.seed}"
    output_base = seed_output_dir / stem
    class_names = np.asarray([DISPLAY_NAMES[name] for name in CLASS_ORDER])
    class_count_array = np.asarray([class_counts[name] for name in CLASS_ORDER], dtype=np.int64)
    np.savez_compressed(
        output_base.with_suffix(".npz"),
        features=features,
        labels=labels,
        tsne=tsne_xy,
        class_names=class_names,
        class_counts=class_count_array,
    )
    scatter_tsne(
        tsne_xy=tsne_xy,
        labels=labels,
        class_counts=class_counts,
        exclude_class=args.exclude_class,
        feature_mode=args.feature_mode,
        output_base=output_base,
    )

    metadata = {
        "feature_mode": args.feature_mode,
        "exclude_class": args.exclude_class,
        "seed": args.seed,
        "ckpt_path": args.ckpt_path,
        "freq_stats_path": args.freq_stats_path,
        "data_root": args.data_root,
        "num_samples_each_class": args.num_samples_each_class,
        "actual_class_counts": class_counts,
        "class_order": CLASS_ORDER,
        "class_names": [DISPLAY_NAMES[name] for name in CLASS_ORDER],
        "sample_indices_path": str(index_path),
        "perplexity_requested": args.perplexity,
        "perplexity_used": perplexity_used,
        "note": metadata_note(args.feature_mode),
    }
    with output_base.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")
    print(f"Saved: {output_base.with_suffix('.npz')}")
    print(f"Saved: {output_base.with_suffix('.json')}")


if __name__ == "__main__":
    main()
