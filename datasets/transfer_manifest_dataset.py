"""Dataset for external transfer manifests; no data work occurs at import time."""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.ddfsd_datasets import make_eval_transform


class TransferManifestDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None):
        self.manifest_path = str(Path(manifest_path).resolve())
        with open(self.manifest_path, newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Transfer manifest is empty: {self.manifest_path}")
        for index, row in enumerate(self.rows):
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid binary label at row {index}: {row.get('label')!r}") from exc
            if label not in (0, 1):
                raise ValueError(f"Label must be 0/1 at row {index}, got {label}")
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(f"Manifest image does not exist at row {index}: {row['image_path']}")
        self.transform = transform or make_eval_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            with Image.open(row["image_path"]) as image:
                image.load()
                image = image.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Corrupt/unreadable transfer image: {row['image_path']}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"]), dict(row)


def transfer_collate_fn(batch):
    images, labels, metadata = zip(*batch)
    return torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.long), list(metadata)
