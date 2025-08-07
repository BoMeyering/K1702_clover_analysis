"""
train_object_detection_model.py
Main Training Script for the Object Detection Model
BoMeyering 2025
"""

import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from src.models import create_effdet_model
from src.datasets import ObjDetDataset
from src.transforms import get_train_obj_transforms, get_val_obj_transforms
from src.trainer import ObjTrainer
from src.utils.loggers import setup_loggers
from src.utils.collate_functions import custom_collate
from pathlib import Path

model_config = {
    "image_size": (512, 512),
    "architecture": "efficientdet_d0",
    "pretrained": True,
    "num_classes": 3,   # including background class
    "max_det_per_image": 50
}

EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints/object_detection_models'
MODEL_RUN_NAME = "_".join([model_config['architecture'], str(model_config['image_size'][0])])

# Instantiate model with config dict
model = create_effdet_model(**model_config)

# Set computational device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Optimizer
optimizer = SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
logging.info(f"Set optimizer {type(optimizer)}")

# LR Scheduler
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
logging.info(f"Set scheduler {type(scheduler)}")

def main():
    logging.info(f"Set scheduler {type(scheduler)}")

    # Instantiate the data augmentations
    train_transforms = get_train_obj_transforms(resize=model_config['image_size'])
    val_transforms = get_val_obj_transforms(resize=model_config['image_size'])
    
    # Instantiate Datasets
    train_ds = ObjDetDataset(transforms=train_transforms)
    val_ds = ObjDetDataset(transforms=val_transforms, split='val')

    # Create Dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate  
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate 
    )

    objdet_trainer = ObjTrainer(
        model_run_name=MODEL_RUN_NAME,
        model=model,
        device=device,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR
    )

    logging.info(f"Created object detection trainer class {type(objdet_trainer)}")

    objdet_trainer.train()

if __name__ == '__main__':
    main()