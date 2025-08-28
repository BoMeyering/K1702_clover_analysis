"""
src.transforms.py
Image Augmentations
BoMeyering 2025
"""

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_seg_transforms(resize: tuple[int, int]):
    """_summary_

    Args:
        resize (tuple[int, int]): _description_
    """

    transforms = A.Compose(
        [
            A.Resize(resize[0], resize[1], p=1.0),
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2),
            A.Rotate(limit=25),
            A.GaussianBlur(p=0.5),
            ToTensorV2()
        ],
        additional_targets={'mask': 'mask'}
    )

    return transforms

def get_train_obj_transforms(resize: tuple[int, int]):
    """_summary_

    Args:
        resize (tuple[int, int]): _description_
    """
    transforms = A.Compose(
        [
            A.Resize(resize[0], resize[1], p=1.0),
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2),
            A.Rotate(limit=25),
            A.GaussianBlur(p=0.5),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        )
    )

    return transforms

def get_val_seg_transforms(resize: tuple[int, int]):
    """_summary_

    Args:
        resize (tuple[int, int]): _description_
    """

    transforms = A.Compose(
        [
            A.Resize(resize[0], resize[1], p=1.0),
            A.Normalize(),
            ToTensorV2()
        ],
        additional_targets={'mask': 'mask'}
    )

    return transforms

def get_val_obj_transforms(resize: tuple[int, int]):
    """_summary_

    Args:
        resize (tuple[int, int]): _description_
    """

    transforms = A.Compose(
        [
            A.Resize(resize[0], resize[1], p=1.0),
            A.Normalize(),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        )
    )

    return transforms