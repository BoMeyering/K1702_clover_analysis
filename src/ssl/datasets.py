"""
src.ssl.datasets.py
Datasets for FlexMatch training on the HQ Labelbox masks.

Images are addressed by img_id (data/processed/images/<img_id>.jpg) so the labeled, unlabeled and
evaluation pools can be carved out of one image directory by id list. JPEGs are decoded at reduced
resolution (cv2.IMREAD_REDUCED_COLOR_n) for speed; targets are resized (nearest) to match.
BoMeyering 2025
"""

from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = 255

_REDUCED_FLAGS = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}


def read_image(path: str | Path, reduction: int = 1) -> np.ndarray:
    """Read an RGB uint8 image, decoded at 1/reduction resolution."""
    if reduction not in _REDUCED_FLAGS:
        raise ValueError(f"reduction must be one of {list(_REDUCED_FLAGS)}")
    img = cv2.imread(str(path), _REDUCED_FLAGS[reduction])
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_target(path: str | Path, shape: Sequence[int]) -> np.ndarray:
    """Read a class-index PNG and resize it (nearest) to (H, W) = shape."""
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise FileNotFoundError(f"Could not read target {path}")
    if target.shape[:2] != tuple(shape[:2]):
        target = cv2.resize(
            target, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return target


class LabeledDataset(Dataset):
    """Images with class-index targets, for training or evaluation."""

    def __init__(
        self,
        img_ids: Sequence[str],
        image_dir: str | Path,
        target_dir: str | Path,
        geometric: A.Compose | None = None,
        photometric: A.Compose | None = None,
        normalize: A.Compose | None = None,
        image_only: A.Compose | None = None,
        reduction: int = 2,
    ):
        """
        Training: pass geometric (+ photometric) and normalize; image and target share the geometry.
        Evaluation: pass image_only (resize + normalise); the target is returned at decoded resolution
        so predictions can be upsampled and scored at full detail.
        """
        self.img_ids = sorted(img_ids)
        self.image_dir = Path(image_dir)
        self.target_dir = Path(target_dir)
        self.geometric, self.photometric, self.normalize, self.image_only = (
            geometric,
            photometric,
            normalize,
            image_only,
        )
        self.reduction = reduction
        if (image_only is None) == (normalize is None):
            raise ValueError(
                "Pass either normalize (training) or image_only (evaluation)"
            )
        missing = [
            i for i in self.img_ids if not (self.target_dir / f"{i}.png").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} targets missing in {self.target_dir}, e.g. {missing[:3]}"
            )

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img = read_image(self.image_dir / f"{img_id}.jpg", self.reduction)
        target = read_target(self.target_dir / f"{img_id}.png", img.shape)

        if self.image_only is not None:
            return (
                self.image_only(image=img)["image"],
                torch.from_numpy(target).long(),
                img_id,
            )

        if self.geometric is not None:
            out = self.geometric(image=img, target=target)
            img, target = out["image"], out["target"]
        if self.photometric is not None:
            img = self.photometric(image=img)["image"]
        out = self.normalize(image=img, target=target)
        return out["image"], out["target"].long(), img_id


class UnlabeledDataset(Dataset):
    """Unlabeled images returned as a pixel-aligned (weak, strong) pair.

    If sam_target_dir is given, the old SAM mask goes through the same geometry and is returned as a
    diagnostic only (scoring pseudo-labels); it is never used as a training target.
    """

    def __init__(
        self,
        img_ids: Sequence[str],
        image_dir: str | Path,
        geometric: A.Compose,
        strong: A.Compose,
        normalize: A.Compose,
        reduction: int = 2,
        sam_target_dir: str | Path | None = None,
    ):
        self.img_ids = sorted(img_ids)
        self.image_dir = Path(image_dir)
        self.geometric, self.strong, self.normalize = geometric, strong, normalize
        self.reduction = reduction
        self.sam_target_dir = Path(sam_target_dir) if sam_target_dir else None

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img = read_image(self.image_dir / f"{img_id}.jpg", self.reduction)

        sam_path = (
            self.sam_target_dir / f"{img_id}.png" if self.sam_target_dir else None
        )
        if sam_path is not None and sam_path.exists():
            out = self.geometric(image=img, target=read_target(sam_path, img.shape))
            img, sam = out["image"], torch.from_numpy(out["target"]).long()
        else:
            img = self.geometric(image=img)["image"]
            sam = torch.full(img.shape[:2], IGNORE_INDEX, dtype=torch.long)

        weak = self.normalize(image=img)["image"]
        strong = self.normalize(image=self.strong(image=img)["image"])["image"]
        return weak, strong, sam, img_id
