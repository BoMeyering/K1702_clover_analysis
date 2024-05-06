from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
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
            num_workers: int=4,
            batch_size:int =8
    ) -> None:
            
        self.train_ds=train_dataset
        self.valid_ds=val_dataset
        self.test_ds=test_dataset
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
        if self.test_ds:
            test_loader=DataLoader(
                dataset=self.test_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers
            )
            return test_loader
        else: 
            raise AttributeError(f"self.test_ds not found. Please instantiate the DataModule with a test dataset.")
    

if __name__ == '__main__':
    from datasets import CloverDataset
    from transformers import SegformerImageProcessor
    from transforms import get_train_transforms, get_val_transforms
    print(True)
    train_img_dir = './data/images/train'
    val_img_dir = './data/images/val'
    test_img_dir = './data/images/test'
    train_mask_dir = './data/annotations/train'
    val_mask_dir = './data/annotations/val'
    test_mask_dir = './data/annotations/test'

    train_dataset = CloverDataset(
        img_dir=train_img_dir, 
        mask_dir=train_mask_dir, 
        img_processor=SegformerImageProcessor(do_reduce_labels=True), 
        transforms=get_train_transforms((1000,1000))
    )
    val_dataset = CloverDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_processor=SegformerImageProcessor(do_reduce_labels=True),
        transforms=get_val_transforms((1000,1000))
    )
    test_dataset = CloverDataset(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_processor=SegformerImageProcessor(do_reduce_labels=True),
        transforms=get_val_transforms((1000,1000))
    )
    sdm = SegformerDataModule(train_dataset=train_dataset,
                              val_dataset=val_dataset,
                            #   test_dataset=test_dataset,
                              num_workers=4, 
                              batch_size=8)
    
    iter_tdl = iter(sdm.test_dataloader())
    while True:
        batch = next(iter_tdl)
        print(batch)

