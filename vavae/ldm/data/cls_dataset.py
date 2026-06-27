from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import os
import csv
import random

from PIL import Image
from PIL import ImageFile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


class_names = [
    "noise",
    "blur",
    "rain",
    "underexposure",
    "haze",
    "reflection",
    "raindrop",
    "snow",
    "overexposure",
    "moire",
    "low",
    "clean",
    "imagenet",
]


def _default_image_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


@dataclass
class PairedRecord:
    input_path: str
    label: int


class PairedImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 256,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        augment: bool = False,
        max_retries: int = 10,
    ) -> None:
        super().__init__()
        self.class_names = [
            "noise",
            "blur",
            "rain",
            "underexposure",
            "haze",
            "reflection",
            "raindrop",
            "snow",
            "overexposure",
            "moire",
            "clean",
            "imagenet",
        ]
        self.csv_path = csv_path
        self.root_dir = os.path.dirname(os.path.abspath(csv_path))
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        self.image_size = (image_size, image_size)
        self.augment = augment
        print(self.augment)
        self.deterministic_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.base_transform = T.CenterCrop(self.image_size)

        if self.augment:
            self.flip_p = 0.5
            self.crop_scale = (0.9, 1.0)
            self.crop_ratio = (0.9, 1.1)

        self.max_retries = max_retries
        self.records: List[PairedRecord] = self._read_csv()

    def _read_csv(self) -> List[PairedRecord]:
        records: List[PairedRecord] = []
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 2:
                    raise ValueError("Each CSV row must have 2 columns: image_path,label")
                image_rel, label_raw = row[0].strip(), row[1].strip()
                image_path = (
                    image_rel
                    if os.path.isabs(image_rel)
                    else os.path.normpath(os.path.join(self.root_dir, image_rel))
                )

                if label_raw.isdigit() or (label_raw.startswith("-") and label_raw[1:].isdigit()):
                    label_idx = int(label_raw)
                else:
                    if label_raw not in self.class_to_idx:
                        raise ValueError(f"Unknown label '{label_raw}'. Expected one of: {self.class_names}")
                    label_idx = self.class_to_idx[label_raw]

                records.append(PairedRecord(input_path=image_path, label=label_idx))
        if not records:
            raise ValueError("No records loaded from CSV. Check the file paths and format.")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: str) -> Optional[Image.Image]:
        try:
            with Image.open(path) as img:
                img.load()
                img = img.convert("RGB")
                return img.copy()
        except (OSError, IOError) as e:
            print(f"[Error] Failed to load image or the file is corrupted: {path}. Error: {e}")
            return None
        except Exception as e:
            print(f"[Error] Unexpected error while loading image: {path}. Error: {e}")
            return None

    def __getitem__(self, index: int) -> dict:
        current_index = index
        retries = 0

        while retries < self.max_retries:
            rec = self.records[current_index]
            img = self._load_image(rec.input_path)

            if img is not None:
                class_label = rec.label
                target_size = self.image_size

                if self.augment:
                    i, j, h, w = T.RandomResizedCrop.get_params(img, self.crop_scale, self.crop_ratio)

                    img = TF.resized_crop(
                        img,
                        i,
                        j,
                        h,
                        w,
                        target_size,
                        T.InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                    if torch.rand(1) < self.flip_p:
                        img = TF.hflip(img)
                else:
                    img = self.base_transform(img)

                img_tensor = self.deterministic_transform(img)

                batch = {
                    "image": img_tensor,
                    "label": class_label,
                    "degradation": self.class_names[rec.label],
                }
                return batch

            retries += 1
            new_index = random.randint(0, len(self) - 1)
            if new_index == current_index:
                current_index = (current_index + 1) % len(self)
            else:
                current_index = new_index

        raise RuntimeError(
            f"Reached the maximum retry count ({self.max_retries}); failed to load a valid sample."
        )


class PairedTrainDataset(PairedImageDataset):
    """Dataset for training with image augmentation enabled."""

    def __init__(self, csv_path: str, image_size: int = 256) -> None:
        super().__init__(csv_path=csv_path, image_size=image_size, augment=True)
        print(f"PairedTrainDataset loaded {len(self)} records with augmentation enabled.")


class PairedTestDataset(PairedImageDataset):
    """Dataset for validation/testing with deterministic center crop."""

    def __init__(self, csv_path: str, image_size: int = 256) -> None:
        super().__init__(csv_path=csv_path, image_size=image_size, augment=False)
        print(f"PairedTestDataset loaded {len(self)} records with augmentation disabled (CenterCrop only).")
