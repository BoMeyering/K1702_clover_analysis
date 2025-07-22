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
from src.transforms import get_train_obj_transforms, get_train_seg_transforms, get_val_obj_transforms, get_val_seg_transforms

class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str='data/processed', split: str='train'):
        self.split = split
        self.data_dir = data_dir

    def __getitem__(self):
        pass

    def __len__(self):
        pass

class ObjDetDataset(Dataset):
    def __init__(
            self, 
            transforms: Compose, 
            data_dir: str='data/processed', 
            split: str='train', 
            img_name_path: str=None, 
            bbox_path: str=None
        ):

        self.split = split
        self.data_dir = data_dir
        self.transforms = transforms(resize=(512, 512))

        self.img_ids = pl.read_csv(img_name_path)\
            .filter(pl.col('split')==self.split)\
            .select('img_id')['img_id']\
            .to_list()
        self.img_ids.sort()
        
        
        self.bboxes = pl.read_csv(bbox_path)\
            .with_columns(
                (pl.col('x1') + pl.col('width')).alias('x2'),
                (pl.col('y1') + pl.col('height')).alias('y2')
            )

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
        # img_path = str(Path(self.data_dir) / 'images' / img_id)
        img_path = str(Path(self.data_dir) / img_id)
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
        targets = augmented['bboxes']
        class_labels = augmented['labels']

        return img, targets, class_labels

    def __len__(self):
        return len(self.img_ids)
    
