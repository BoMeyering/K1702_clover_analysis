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
from src.models import create_fasterrcnn_model
from src.datasets import ObjDetDataset
from src.transforms import get_train_obj_transforms, get_val_obj_transforms
from src.trainer import ObjTrainer
from src.utils.loggers import setup_loggers
from src.utils.collate_functions import custom_collate
from pathlib import Path

model_config = {
    "architecture": "fasterrcnn_resnet50_fpn",
    "pretrained": True,
    "num_classes": 2,   # 1 class + background
    "max_det_per_image": 20,
    "image_size": (1024, 1024)  # handled in transforms, not the model
}

EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints/object_detection_models'
MODEL_RUN_NAME = "_".join([model_config['architecture'], str(model_config['image_size'][0])])

setup_loggers(model_run=MODEL_RUN_NAME, log_dir='logs', log_level='INFO')

logger = logging.getLogger()

# Set computational device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f"Set computational device: {device}")


# Instantiate model with config dict
model = create_fasterrcnn_model(**model_config)
model = model.to(device)
logger.info(f"Instantiated object detection model {type(model)} and moved to {device}.")


# Optimizer
optimizer = SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
logger.info(f"Set optimizer {type(optimizer)}")


# LR Scheduler
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
logger.info(f"Set scheduler {type(scheduler)}")

def main():

    # Instantiate the data augmentations
    train_transforms = get_train_obj_transforms(resize=model_config['image_size'])
    val_transforms = get_val_obj_transforms(resize=model_config['image_size'])
    
    # Instantiate Datasets
    train_ds = ObjDetDataset(transforms=train_transforms)
    val_ds = ObjDetDataset(transforms=val_transforms, split='val')

    # Create Dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=custom_collate  
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=4,
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

    logger.info(f"Created model trainer class {type(objdet_trainer)}")

    objdet_trainer.train()

if __name__ == '__main__':
    main()