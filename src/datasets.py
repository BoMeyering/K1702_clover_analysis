"""
src.datasets.py
Dataset classes with built-in dataloader + sampler
BoMeyering 2025
"""

import torch
import json
import cv2
import polars as pl
from albumentations import Compose
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Union, Tuple, Optional
import src.transforms
from src.utils.collate_functions import custom_collate, seg_collate



class SegmentationDataset(Dataset):
    def __init__(
        self,
        transforms: Compose,
        data_dir: Union[Path, str] = 'data',
        split: str = 'train',
        img_resize: Tuple = (512, 512)
    ):
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'processed' / 'images'
        self.target_dir = self.data_dir / 'processed' / 'targets'
        self.transforms = transforms

        self.img_ids = pl.read_csv(self.data_dir / 'data_split.csv') \
            .filter(pl.col('split') == self.split) \
            .select('img_id')['img_id'] \
            .to_list()
        self.img_ids.sort()

        with open('metadata/segmentation_class_map.json', 'r') as f:
            self.mapping = json.load(f)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img_path = str(self.img_dir / (img_id + '.jpg'))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        target_path = str(self.target_dir / (img_id + '.png'))
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        augmented = self.transforms(image=img, mask=target)
        img = augmented['image']
        target = augmented['mask']

        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)

    def get_dataloader(
        self,
        batch_size: int = 4,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 0,           # added
        pin_memory: bool = True         # added
    ) -> DataLoader:
        sampler = None
        if distributed:
            sampler = DistributedSampler(self, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
            shuffle = False  # must disable shuffle when using sampler
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,         # pass to DataLoader
            pin_memory=pin_memory,           # pass to DataLoader
            persistent_workers=(num_workers > 0),
            collate_fn=seg_collate
        )


class ObjDetDataset(Dataset):
    def __init__(
        self,
        transforms: Compose,
        data_dir: Union[Path, str] = 'data',
        split: str = 'train'
    ):
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'processed' / 'images'
        self.transforms = transforms

        self.img_ids = pl.read_csv(self.data_dir / 'data_split.csv') \
            .filter(pl.col('split') == self.split) \
            .select('img_id')['img_id'] \
            .to_list()
        self.img_ids.sort()

        self.bboxes = pl.read_csv(self.data_dir / 'bboxes.csv')

        with open('metadata/obj_det_class_map.json', 'r') as f:
            self.mapping = json.load(f)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        img_path = str(self.img_dir / (img_id + '.jpg'))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        bboxes_df = self.bboxes.filter(
            (pl.col('img_id') == img_id) & (pl.col('class') == 'quadrat_corner')
        )

        bboxes = bboxes_df.select(['x1', 'y1', 'x2', 'y2']).to_numpy()
        labels = [self.mapping['quadrat_corner']] * len(bboxes)

        augmented = self.transforms(image=img, bboxes=bboxes, labels=labels)
        img = augmented['image']
        bboxes = augmented['bboxes']
        labels = [int(x) for x in augmented['labels']]

        _, new_h, new_w = img.shape
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": bboxes, "labels": labels}
        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)

    def get_dataloader(
        self,
        batch_size: int = 4,
        distributed: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> DataLoader:
        sampler = None
        if distributed:
            sampler = DistributedSampler(self, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
            shuffle = False
        return DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            collate_fn=custom_collate
        )
