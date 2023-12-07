import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms(img_size: tuple[int, int]):
    """
    Training augmentations for images and masks

    img_size: a tuple of ints passed to albumentations.resize

    return -> an instantiated image transformation function
    """
    transforms = A.Compose(
        [
            A.Resize(img_size[1], img_size[0], always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.Flip(p=0.5),
            A.Rotate(limit=25),
            ToTensorV2(always_apply=True)
        ],
        is_check_shapes=True
    )

    return transforms

def get_val_transforms(img_size: tuple[int, int]):
    """
    Validation augmentations for images and masks

    img_size: a tuple of ints passed to albumentations.resize

    return -> an instantiated image transformation function    
    """
    transforms = A.Compose(
        [
            A.Resize(img_size[1], img_size[0], always_apply=True),
            ToTensorV2(always_apply=True)
        ],
        is_check_shapes=True
    )

    return transforms
