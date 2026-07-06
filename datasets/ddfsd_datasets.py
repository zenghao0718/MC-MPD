"""Datasets and sampling helpers for DDFSD."""

import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from torchvision import transforms


ALL_CLASSES = ["real", "ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
FAKE_CLASSES = ["ADM", "BigGAN", "glide", "Midjourney", "SD", "VQDM"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_generator_name(class_name: str, allow_real: bool = False) -> None:
    valid = ALL_CLASSES if allow_real else FAKE_CLASSES
    if class_name not in valid:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of: {valid}")


def get_train_fake_classes(exclude_class: str) -> List[str]:
    validate_generator_name(exclude_class)
    return [name for name in FAKE_CLASSES if name != exclude_class]


def _collect_image_paths(root: str) -> List[str]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    paths = [
        str(path)
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
    ]
    paths.sort()
    if not paths:
        raise ValueError(f"No image files found under: {root}")
    return paths


class ImagePathDataset(Dataset):
    """Image dataset that treats every image under root as one class source."""

    def __init__(self, root: str, transform=None):
        self.root = root
        self.paths = _collect_image_paths(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


def make_train_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )


def make_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def make_stats_transform():
    return make_eval_transform()


def setup_ddfsd_infinite_train_dataloader(
    folder_path: str,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = True,
):
    dataset = ImagePathDataset(folder_path, transform=make_train_transform())
    sampler = DistributedSampler(dataset) if dist.is_available() and dist.is_initialized() else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        yield from loader


def load_ddfsd_class_dataset(data_root: str, class_name: str, split: str, transform=None) -> ImagePathDataset:
    validate_generator_name(class_name, allow_real=True)
    if split not in {"train", "val"}:
        raise ValueError(f"DDFSD uses only train/val splits, got: {split}")
    if transform is None:
        transform = make_eval_transform()
    return ImagePathDataset(os.path.join(data_root, class_name, split), transform=transform)


def make_subset_loader(
    dataset: Dataset,
    indices: Sequence[int],
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def sample_support_query_indices(
    dataset_size: int,
    support_shot: int,
    seed: int,
    max_query: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    if dataset_size <= support_shot:
        raise ValueError(
            f"Need more than {support_shot} images to build support/query, got {dataset_size}."
        )

    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    support = indices[:support_shot]
    query = indices[support_shot:]
    if max_query is not None and max_query > 0:
        query = query[:max_query]
    return support, query


def build_train_iterators(
    data_root: str,
    classes: Iterable[str],
    images_per_class_per_step: int,
    num_workers: int,
    pin_memory: bool = True,
):
    return {
        class_name: setup_ddfsd_infinite_train_dataloader(
            folder_path=os.path.join(data_root, class_name, "train"),
            batch_size=images_per_class_per_step,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        for class_name in classes
    }
