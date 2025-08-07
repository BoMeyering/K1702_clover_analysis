"""
src/utils/collate_functions.py
"""

import torch

def custom_collate(batch):
    imgs, targets, img_ids = tuple(zip(*batch))

    imgs = torch.stack(imgs)
    imgs = imgs.float()

    # print(f"RAW TARGETS: {targets}")

    bboxes = [target["bboxes"].float() for target in targets]
    labels = [target["labels"].float() for target in targets]
    img_size = torch.tensor([target["img_size"] for target in targets]).float()
    img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

    targets = {
        "bbox": bboxes,
        "cls": labels,
        "img_size": img_size,
        "img_scale": img_scale
    }

    # print(f"NEW TARGETS: {targets}")
    return imgs, targets, img_ids