from torch.utils.data import Dataset
from glob import glob
from typing import Any
import os
import cv2
from scipy.io import loadmat
from transformers import SegformerFeatureExtractor
from transformers import SegformerImageProcessor
import torch

class CloverDataset(Dataset):
    """
    A subclass of a Pytorch Dataset


    """
    def __init__(
            self,
            img_dir,
            mask_dir,
            img_processor,
            transforms
    ) -> None:
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_processor = img_processor
        self.transforms = transforms
        self.img_names = sorted(
            glob(
                pathname="*.jpg",
                root_dir=img_dir
            )
        )

        self.img_paths = sorted(
            glob(
                pathname=os.path.join(self.img_dir, "*.jpg")
            )
        )

        self.mask_names = sorted(
            glob(
                pathname="*.mat",
                root_dir=self.mask_dir
            )
        )
        self.mask_paths = sorted(
            glob(
                pathname=os.path.join(self.mask_dir, "*.mat")
            )
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index) -> Any:
        img = cv2.imread(self.img_paths[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        mask_dict = loadmat(self.mask_paths[index])
        mask = mask_dict['data']

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image'].astype('uint8')
        mask = transformed['mask']

        processed = self.img_processor(img, mask, return_tensors='pt')
        
        return processed
    

if __name__ == '__main__':
    from transforms import get_train_transforms
    new_ds = CloverDataset(img_dir='./data/images/train',
                           mask_dir='./data/annotations/train',
                           img_processor=SegformerImageProcessor(do_reduce_labels=True),
                           transforms=get_train_transforms((1000, 1000)))
    print(len(new_ds.img_names))
    print(len(new_ds.img_paths))
    for i in range(len(new_ds.img_names)):
        embed = new_ds[i]['labels']
        print(torch.unique(embed))

        pass