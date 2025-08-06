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
from pathlib import Path

model_config = {
    "architecture": "EfficientDet",  # or your detection model architecture
    "backbone_name": "efficientdet_d0",
    "pretrained": True,
    "num_classes": 3   # including background class
}

EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints/object_detection_models'
IMG_RESIZE = (1024, 1024)
MODEL_RUN_NAME = "_".join([model_config['architecture'], model_config['backbone_name'], str(IMG_RESIZE[0])])


def custom_collate(batch):
    images, targets, image_ids = tuple(zip(*batch))
    images = torch.stack(images)
    images = images.float()

    boxes = [target["bboxes"].float() for target in targets]
    labels = [target["cls"].float() for target in targets]
    img_size = [target["img_size"].float() for target in targets]
    img_scale = [target["img_scale"].float() for target in targets]

    annotations = {
        "bbox": boxes,
        "cls": labels,
        "img_size": img_size,
        "img_scale": img_scale,
    }

    return images, annotations, image_ids

model = create_effdet_model(image_size=IMG_RESIZE)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Optimizer
optimizer = SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# logging.info(f"Set optimizer {type(optimizer)}")

# LR Scheduler
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
# logging.info(f"Set scheduler {type(scheduler)}")

def main():
    # logging.info(f"Set scheduler {type(scheduler)}")

    train_ds = ObjDetDataset(
        transforms=get_train_obj_transforms,
        img_resize = IMG_RESIZE, 
    
    )

    val_ds = ObjDetDataset(
        transforms=get_val_obj_transforms,
        img_resize = IMG_RESIZE,
        split='val',
    )

    # Create Dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate  
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
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

    # logging.info(f"Created object detection trainer class {type(objdet_trainer)}")

    objdet_trainer.train()

    # Example inference on validation set (optional)
    # model.eval()
    # for idx in range(len(val_ds)):
    #     img, target, img_id = val_ds[idx]
    #     with torch.no_grad():
    #         prediction = model([img.to(device)])
    #     print(f"Prediction for {img_id}: {prediction}")
    #     # Save or process predictions as needed

if __name__ == '__main__':
    main()
