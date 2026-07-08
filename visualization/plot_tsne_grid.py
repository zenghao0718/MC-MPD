"""Build DDFSD t-SNE grid figures from saved NPZ/JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
DEFAULT_EXCLUDES = ["Midjourney", "glide", "ADM", "SD", "VQDM", "BigGAN"]
DEFAULT_FEATURE_MODES = ["rgb", "freq", "add", "concat"]


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DDFSD t-SNE summary grids")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory containing exclude_<class>/visualization/seed<seed>/ files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for grid figures. Defaults to <run_dir>/visualization/seed<seed>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_classes", type=str, default=",".join(DEFAULT_EXCLUDES))
    parser.add_argument("--feature_modes", type=str, default=",".join(DEFAULT_FEATURE_MODES))
    parser.add_argument("--point_size", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--grid_type",
        type=str,
        default="both",
        choices=["both", "four_modes", "all_excludes"],
        help=(
            "'four_modes' only builds the per-exclude-class 2x2 (rgb/freq/add/concat) grids. "
            "'all_excludes' only builds the per-feature-mode 2x3 grids across --exclude_classes. "
            "'both' (default) builds both. Use 'four_modes' when --exclude_classes is a partial "
            "subset of the 6 classes to avoid producing a misleading incomplete all-excludes grid."
        ),
    )
    return parser.parse_args()


def artifact_paths(run_dir: Path, exclude_class: str, feature_mode: str, seed: int):
    seed_dir = run_dir / f"exclude_{exclude_class}" / "visualization" / f"seed{seed}"
    stem = f"tsne_{feature_mode}_exclude_{exclude_class}_seed{seed}"
    return seed_dir / f"{stem}.npz", seed_dir / f"{stem}.json"


def preflight(run_dir: Path, exclude_classes: List[str], feature_modes: List[str], seed: int) -> None:
    missing = []
    for exclude_class in exclude_classes:
        for feature_mode in feature_modes:
            npz_path, json_path = artifact_paths(run_dir, exclude_class, feature_mode, seed)
            if not npz_path.is_file():
                missing.append(str(npz_path))
            if not json_path.is_file():
                missing.append(str(json_path))
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required t-SNE NPZ/JSON artifacts; refusing to create incomplete grids:\n"
            f"{missing_text}"
        )


def load_artifact(run_dir: Path, exclude_class: str, feature_mode: str, seed: int):
    npz_path, json_path = artifact_paths(run_dir, exclude_class, feature_mode, seed)
    with np.load(npz_path) as data:
        tsne_xy = np.asarray(data["tsne"])
        labels = np.asarray(data["labels"])
        class_counts = np.asarray(data["class_counts"]).astype(int)
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "tsne": tsne_xy,
        "labels": labels,
        "class_counts": class_counts,
        "metadata": metadata,
        "npz_path": npz_path,
        "json_path": json_path,
    }


def load_all(run_dir: Path, exclude_classes: List[str], feature_modes: List[str], seed: int):
    artifacts = {}
    for exclude_class in exclude_classes:
        for feature_mode in feature_modes:
            artifacts[(exclude_class, feature_mode)] = load_artifact(
                run_dir, exclude_class, feature_mode, seed
            )
    return artifacts


def scatter_subplot(ax, artifact, exclude_class: str, point_size: float, title: str) -> None:
    tsne_xy = artifact["tsne"]
    labels = artifact["labels"]
    class_counts = artifact["class_counts"]
    for label, class_name in enumerate(CLASS_ORDER):
        mask = labels == label
        legend_name = DISPLAY_NAMES[class_name]
        count = int(class_counts[label])
        if class_name == exclude_class:
            legend_name = f"{legend_name} (test, n={count})"
        else:
            legend_name = f"{legend_name} (n={count})"
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            c=COLORS[class_name],
            s=point_size,
            alpha=0.82,
            linewidths=0,
            label=legend_name,
        )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=5.6, title="Classes", title_fontsize=6.2, frameon=True)


def save_figure(fig, output_base: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_base.with_suffix('.png')}")
    print(f"Saved: {output_base.with_suffix('.pdf')}")


def plot_mode_grid(
    artifacts: Dict[Tuple[str, str], dict],
    exclude_class: str,
    feature_modes: List[str],
    seed: int,
    output_dir: Path,
    point_size: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=dpi)
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, panel, feature_mode in zip(axes.flat, labels, feature_modes):
        title = (
            f"{panel} {FEATURE_LABELS[feature_mode]} | "
            f"Test: {DISPLAY_NAMES[exclude_class]}"
        )
        scatter_subplot(
            ax,
            artifacts[(exclude_class, feature_mode)],
            exclude_class,
            point_size,
            title,
        )
    output_base = output_dir / f"exclude_{exclude_class}_4mode_grid_seed{seed}"
    save_figure(fig, output_base)


def plot_exclude_grid(
    artifacts: Dict[Tuple[str, str], dict],
    feature_mode: str,
    exclude_classes: List[str],
    seed: int,
    output_dir: Path,
    point_size: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=dpi)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, panel, exclude_class in zip(axes.flat, labels, exclude_classes):
        title = f"{panel} Test on {DISPLAY_NAMES[exclude_class]}"
        scatter_subplot(
            ax,
            artifacts[(exclude_class, feature_mode)],
            exclude_class,
            point_size,
            title,
        )
    output_base = output_dir / f"{feature_mode}_all_excludes_grid_seed{seed}"
    save_figure(fig, output_base)


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualization" / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude_classes = parse_csv(args.exclude_classes)
    feature_modes = parse_csv(args.feature_modes)
    invalid_excludes = [name for name in exclude_classes if name not in DEFAULT_EXCLUDES]
    invalid_modes = [name for name in feature_modes if name not in DEFAULT_FEATURE_MODES]
    if invalid_excludes:
        raise ValueError(f"Unsupported exclude_classes: {invalid_excludes}")
    if invalid_modes:
        raise ValueError(f"Unsupported feature_modes: {invalid_modes}")

    preflight(run_dir, exclude_classes, feature_modes, args.seed)
    artifacts = load_all(run_dir, exclude_classes, feature_modes, args.seed)

    if args.grid_type in {"both", "four_modes"}:
        for exclude_class in exclude_classes:
            plot_mode_grid(
                artifacts,
                exclude_class,
                feature_modes,
                args.seed,
                output_dir,
                args.point_size,
                args.dpi,
            )
    if args.grid_type in {"both", "all_excludes"}:
        if len(exclude_classes) < len(DEFAULT_EXCLUDES):
            print(
                "Skipping all_excludes grids: --exclude_classes is a partial subset "
                f"({exclude_classes}) of the full class list ({DEFAULT_EXCLUDES}); "
                "an incomplete all_excludes grid would misrepresent results. "
                "Pass --grid_type four_modes explicitly to silence this note."
            )
        else:
            for feature_mode in feature_modes:
                plot_exclude_grid(
                    artifacts,
                    feature_mode,
                    exclude_classes,
                    args.seed,
                    output_dir,
                    args.point_size,
                    args.dpi,
                )


if __name__ == "__main__":
    main()
