"""
src.datasets.py
Dataset classes
BoMeyering 2025
"""

import torch
import json
import cv2
import polars as pl
from albumentations import Compose
from pathlib import Path
from torch.utils.data import Dataset
from typing import Union, Tuple
from src.transforms import get_train_obj_transforms, get_train_seg_transforms, get_val_obj_transforms, get_val_seg_transforms

class SegmentationDataset(Dataset):
    def __init__(
        self,
        transforms: Compose,
        data_dir: Union[Path, str]='data',
        split: str='train',
        img_resize: Tuple=(512, 512)
    ):
    
        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'processed' / 'images'
        self.target_dir = self.data_dir / 'processed' / 'targets'
        self.transforms = transforms

        # Grab img ids
        self.img_ids = pl.read_csv(self.data_dir / 'data_split.csv')\
            .filter(pl.col('split')==self.split)\
            .select('img_id')['img_id']\
            .to_list()
        self.img_ids.sort()

        # Read in class map
        with open('metadata/segmentation_class_map.json', 'r') as f:
            self.mapping = json.load(f)

    def __getitem__(
        self, 
        index: int
    ):
        # Grab the img_id
        img_id = self.img_ids[index]

        # Construct the image path and read in the image in RGB
        img_path = str(self.img_dir / (img_id + '.jpg'))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        # Construct the mask path and read in the mask
        target_path = str(self.target_dir / (img_id + '.png'))
        target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        # Perform augmentations
        augmented = self.transforms(
            image=img,
            mask=target
        )

        # Grab the transformed images, bounding boxes and labels
        img = augmented['image']
        target = augmented['mask']

        return img, target, img_id

    def __len__(self):
        return len(self.img_ids)

class ObjDetDataset(Dataset):
    def __init__(
            self, 
            transforms: Compose,
            data_dir: Union[Path, str]='data',
            split: str='train'
        ):

        self.split = split
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'processed' / 'images'
        # self.target_dir = self.data_dir / 'processed' / 'targets'
        self.transforms = transforms

        # Grab img ids
        self.img_ids = pl.read_csv(self.data_dir / 'data_split.csv')\
            .filter(pl.col('split')==self.split)\
            .select('img_id')['img_id']\
            .to_list()
        self.img_ids.sort()

        # Grab all bounding boxes
        self.bboxes = pl.read_csv(self.data_dir / 'bboxes.csv')

        # Read in class map
        with open('metadata/obj_det_class_map.json', 'r') as f:
            self.mapping = json.load(f)

    def __getitem__(
            self, 
            index: int
        ):
        
        # Grab the img_id
        img_id = self.img_ids[index]

        # Construct the image path and read in the image in RGB
        img_path = str(self.img_dir / (img_id + '.jpg'))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        # Grab the bboxes for the img_id
        bboxes = self.bboxes.filter(pl.col('img_id')==img_id)\
            .select(['x1', 'y1', 'x2', 'y2'])\
            .to_numpy()
        
        # Grab the class labels and map to integers
        labels = self.bboxes.filter(pl.col('img_id')==img_id)\
            .select('class')['class']\
            .to_list()
        labels = list(map(lambda x: self.mapping.get(x), labels))

        # Perform augmentations
        augmented = self.transforms(
            image=img,
            bboxes=bboxes,
            labels=labels
        )

        # Grab the transformed images, bounding boxes and labels
        img = augmented['image']
        bboxes = augmented['bboxes']
        labels = [int(x) for x in augmented['labels']]

        _, new_h, new_w = img.shape
        bboxes = torch.as_tensor(bboxes ,dtype = torch.float32)
        labels = torch.as_tensor(labels)
        
        target = {
            "bboxes": bboxes,
            "labels": labels,
            "img_size": (new_h, new_w),
            "img_scale": [1.0],
        }

        return img, target, img_id
    
    def __len__(self):
        return len(self.img_ids)
    
