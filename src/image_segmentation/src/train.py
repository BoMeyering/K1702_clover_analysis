import os
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from transformers import SegformerImageProcessor

from transforms import get_train_transforms, get_val_transforms
from datasets import CloverDataset
from data_module import SegformerDataModule
from model_module import SegformerModel
from model import create_model
# Configurations
NUM_CLASSES = 3
IMG_SIZE=(800,800)
DEVICE = 'cuda'

# Set directories
TRAIN_IMG_DIR = "./data/images/train"
VAL_IMG_DIR = "./data/images/val"
TEST_IMG_DIR = "./data/images/test"

TRAIN_MASK_DIR = "./data/annotations/train"
VAL_MASK_DIR = "./data/annotations/val"
TEST_MASK_DIR = "./data/annotations/test"

# Create datasets
train_dataset = CloverDataset(
    img_dir=TRAIN_IMG_DIR,
    mask_dir=TRAIN_MASK_DIR,
    img_processor=SegformerImageProcessor(
        do_reduce_labels=True
    ),
    transforms=get_train_transforms(
        img_size=IMG_SIZE
    )
)

val_dataset = CloverDataset(
    img_dir=VAL_IMG_DIR,
    mask_dir=VAL_MASK_DIR,
    img_processor=SegformerImageProcessor(
        do_reduce_labels=True
    ),
    transforms=get_val_transforms(
        img_size=IMG_SIZE
    )
)

test_dataset = CloverDataset(
    img_dir=TEST_IMG_DIR,
    mask_dir=TEST_MASK_DIR,
    img_processor=SegformerImageProcessor(
        do_reduce_labels=True
    ),
    transforms=get_val_transforms(
        img_size=IMG_SIZE
    )
)

# Create the segformer data module
sdm = SegformerDataModule(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    num_workers=4,
    batch_size=2
)

# Create the segformer model module
seg_mod = SegformerModel(
    num_classes=NUM_CLASSES,
    img_size=IMG_SIZE,
    model_variant='nvidia/mit-b5'
)

# Create a Tensorboard Logger
logger = TensorBoardLogger("lightning_logs", name="my_model")

# Create the lightining trainer
trainer = Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=100,
    logger=logger
)

trainer.fit(seg_mod, sdm)
