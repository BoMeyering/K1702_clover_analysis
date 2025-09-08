"""
src/utils/collate_functions.py
"""

import torch

def custom_collate(batch):
    return tuple(zip(*batch))

def seg_collate(batch):
    imgs, targets, img_ids = zip(*batch)
    imgs = torch.stack(imgs)       # [B, C, H, W]
    targets = torch.stack(targets) # [B, H, W]
    return imgs, targets, img_ids