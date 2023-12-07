from torch.utils.data import Dataset
from glob import glob
from typing import Any
import os
import cv2

class CloverDataset(Dataset):
    """
    A subclass of a Pytorch Dataset


    """
    def __init__(
            self,
            img_dir,
            annotation_dir, 
            feature_extractor=transformers.SegformerFeatureExtractor
    ) -> None:
        
        self.img_names = sorted(
            glob(
                pathname="*.jpg",
                root_dir=img_dir
            )
        )

        self.img_paths = sorted([
            os.path.join(img_dir, img_name)\
            for img_name in self.img_names
        ])

        self.annotation_files = sorted(glob(
            pathname="*.mat",
            root_dir=annotation_dir
        ))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index) -> Any:
        img = cv2.imread(self.img_paths[index])

        return super().__getitem__(index)
    

if __name__ == '__main__':
    new_ds = CloverDataset(img_dir='./data/images/train',
                           annotation_dir='./data/annotations/mat_files')
    print(len(new_ds.img_names))
    print(len(new_ds.img_paths))
    for i in range(880):
        assert new_ds.img_names[i] != new_ds.img_paths[i].split('/')[-1]
    print(len(new_ds.annotation_files))