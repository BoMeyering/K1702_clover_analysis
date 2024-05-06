# Imports
import pandas as pd
import os
from pytorch_lightning import Trainer
from typing import List

from dataset_adaptor import QuadratDatasetAdaptor
from effdet_datamodule import EfficientDetDataModule
from effdet_model import EfficientDetModel
from create_model import create_model

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Settings

IMG_DIR = "./data/images/"
TRAIN_DIR = './data/images/train'
VAL_DIR = './data/images/val'
TEST_DIR = './data/images/test'
AN_PATH = "./data/annotations/bounding_boxes.csv"



# Read in and convert bounding box coords
box_df = pd.read_csv(AN_PATH, index_col=False)
box_df.rename(columns={'left': 'xmin', 'top': 'ymin'}, inplace=True)
box_df['xmax'] = box_df['xmin'] + box_df['width']
box_df['ymax'] = box_df['ymin'] + box_df['height']

box_df = box_df[['img_id', 'class', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax']]\
    .loc[box_df['class'] == 'quadrat_corner']

train_val_df = pd.read_csv(os.path.join(IMG_DIR, 'data_split.csv'))[['img_id', 'split']]

box_df = pd.merge(
    left=train_val_df,
    right=box_df,
    on='img_id',
    how='left'
)

# Subset training and validation dataframes
train_df = box_df.loc[box_df['split'] == 'train']
val_df = box_df.loc[box_df['split'] == 'val']
test_df = box_df.loc[box_df['split'] == 'test']

# Create dataset adaptors
train_ds = QuadratDatasetAdaptor(images_dir_path=TRAIN_DIR, annotations_dataframe=train_df)
val_ds = QuadratDatasetAdaptor(images_dir_path=VAL_DIR, annotations_dataframe=val_df)
test_ds = QuadratDatasetAdaptor(images_dir_path=TEST_DIR, annotations_dataframe=test_df)

dm = EfficientDetDataModule(train_dataset_adaptor=train_ds, 
        validation_dataset_adaptor=val_ds,
        num_workers=4,
        batch_size=2)

model = EfficientDetModel(
    num_classes=1,
    img_size=512
    )

# Set up logger
logger = TensorBoardLogger("./src/quadrat_detection/runs", name=f"quad_det_")

# Callbacks
lr_monitor = LearningRateMonitor(logging_interval='epoch')

checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_loss",
    mode="min",
    filename="quadrat_effdet-{epoch:02d}-{val_loss:.3f}",
    save_on_train_epoch_end=True
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,
    mode='min'
)

# Instantiate trainer and fit model
trainer = Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=30, num_sanity_val_steps=1,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback, early_stopping]
    )

trainer.fit(model, dm)