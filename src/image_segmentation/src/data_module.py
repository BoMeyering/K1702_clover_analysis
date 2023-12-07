from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule
# from datasets import 
from typing import Optional



class SegformerDataModule(LightningDataModule):
    """
    A LightningDataModule to hold all training, validation, and testing Datasets and Dataloaders
    
    """

    def __init__(
            self, 
            train_dataset: Dataset, 
            val_dataset: Dataset, 
            test_dataset: Optional[Dataset] = None,
            train_transforms: albumentations.Compose,
            val_transforms, 
            num_workers=4,
            batch_size=8
    ) -> None:
            
        self.train_ds=train_dataset
        self.valid_ds=val_dataset
        self.test_ds=test_dataset
        self.train_transfrms=train_transforms
        self.val_transforms=val_transforms
        self.num_workers=num_workers
        self.batch_size=batch_size
        super().__init__()
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_loader=DataLoader(
            dataset=self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_loader=DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers
        )
        return val_loader
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset:
            test_loader=DataLoader(
                dataset=self.test_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers
            )
            return test_loader
        else: 
            raise AttributeError(f"self.test_dataset not found. Please instantiate the DataModule with a test dataset.")
    
