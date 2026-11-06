"""
src.ssl.transforms.py
Augmentation pipelines for FlexMatch segmentation.

Unlabeled images go through ONE shared geometric transform; the weak and strong views are both built
from that geometric view, so a pseudo-label predicted on the weak view is pixel-aligned with the strong
view (CutMix is applied on the GPU afterwards). Strong augmentation is photometric, applied to uint8
images before normalisation.

Inputs are full images (3:2 aspect); outputs are squashed to `resize`, matching the inference pipeline
(src/transforms.get_inf_transforms resizes the whole image to 1024x1024).
BoMeyering 2025
"""

from typing import Iterable, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEANS = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RandomResizedCrop aspect-ratio range (crop w / h) centred on the 3:2 field images.
CROP_RATIO = (1.2, 1.8)


def get_geometric_transforms(resize: Tuple[int, int], scale: Tuple[float, float] = (0.35, 1.0)) -> A.Compose:
    """Shared geometry for labeled images and for both views of unlabeled images (uint8 in, uint8 out)."""
    return A.Compose([
        A.RandomResizedCrop(size=tuple(resize), scale=scale, ratio=CROP_RATIO, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], additional_targets={'target': 'mask'})


def get_labeled_photometric_transforms() -> A.Compose:
    """Mild colour augmentation for the labeled images (uint8 in, uint8 out)."""
    return A.Compose([
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def get_strong_photometric_transforms() -> A.Compose:
    """Strong augmentation for the unlabeled strong view (uint8 in, uint8 out)."""
    return A.Compose([
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(0.1, 2.0), p=0.5),
        A.OneOf([
            A.GlassBlur(sigma=0.3, max_delta=2, iterations=1),   # pgcview strong set
            A.ChannelShuffle(),
            A.RGBShift(),
            A.Posterize(num_bits=(4, 6)),
            A.Equalize(),
            A.Sharpen(),
            A.ImageCompression(quality_range=(30, 80)),
            A.GaussNoise(std_range=(0.02, 0.08)),
        ], p=0.8),
    ])


def get_normalize_transforms(means: Iterable = IMAGENET_MEANS, std: Iterable = IMAGENET_STD) -> A.Compose:
    """uint8 HWC -> normalised float CHW tensor (masks, if given, are converted to tensors too)."""
    return A.Compose([A.Normalize(mean=means, std=std), ToTensorV2()], additional_targets={'target': 'mask'})


def get_val_transforms(resize: Tuple[int, int], means: Iterable = IMAGENET_MEANS, std: Iterable = IMAGENET_STD) -> A.Compose:
    """Validation / inference: resize the whole image to the model size and normalise (image only)."""
    return A.Compose([A.Resize(*resize, p=1.0), A.Normalize(mean=means, std=std), ToTensorV2()])
