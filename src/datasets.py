"""
src.datasets.py
Dataset classes
BoMeyering 2025
"""

import torch
import cv2
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, data_dir: str='data/processed', split: str='train'):
        self.split = split
        self.data_dir = data_dir

    def __getitem__(self):
        pass

    def __len__(self):
        pass

class ObjDetDataset(Dataset):
    def __init__(self, data_dir: str='data/processed', split: str='train', img_name_path: str=None, bbox_path: str=None):
        self.split = split
        self.data_dir = data_dir

        self.img_ids = pl.read_csv(img_name_path).\
            filter(pl.col('split')==self.split).\
            select('img_id')['img_id'].to_list().sort()
        
        self.bboxes = pl.read_csv(bbox_path).\
            with_columns(
                (pl.col('x1') + pl.col('width')).alias('x2'),
                (pl.col('y1') + pl.col('height')).alias('y2')
            )

    def __getitem__(self, index: int):
        
        # Grab the img_id
        img_id = self.img_ids[index]

        # Construct the image path and read in the image in RGB
        img_path = str(Path(self.data_dir) / 'images' / img_id)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        # Grab the 
        

    def __len__(self):
        return len(self.img_ids)