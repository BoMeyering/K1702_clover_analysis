"""
train_segmentation_model.py
Main Training Script for the Segmentation Model
BoMeyering 2025
"""

import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from src.datasets import SegmentationDataset, ObjDetDataset
from src.transforms import get_train_seg_transforms, get_val_seg_transforms, get_train_obj_transforms
from src.trainer import SegTrainer
from src.utils.loggers import setup_loggers
from src.models import create_smp_model

import cv2
import torch.nn.functional as F
from torch import argmax
from pathlib import Path

model_config = {
    "architecture": "Segformer",
    "encoder_name": "mit_b1",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "input_channels": 3,
    "classes": 3
}

EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints/segmentation_models'
IMG_RESIZE = (1024, 1024)
MODEL_RUN_NAME = "_".join([model_config['architecture'], model_config['encoder_name'], str(IMG_RESIZE[0])])

# setup_loggers(model_run=MODEL_RUN_NAME, log_dir='logs', log_level='INFO')

# logger = logging.getLogger()

def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # logger.info(f"Set computational device: {device}")

    # Create model
    model = create_smp_model(config=model_config)
    model = model.to(device)
    # logger.info(f"Instantiated segmentation model {type(model)} and sent to computational device {device}.")

    # Optimizer
    optimizer = SGD(params=model.parameters(),momentum=0.9, nesterov=True)
    # logger.info(f"Set optimizer {type(optimizer)}")

    # LR Scheduler
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
    # logger.info(f"Set scheduler {type(scheduler)}")

    # Set loss function
    criterion = CrossEntropyLoss()
    # logger.info(f"Set loss criterion {type(criterion)}")

    # Create datasets
    train_ds = SegmentationDataset(
        transforms=get_train_seg_transforms,
        img_resize=IMG_RESIZE
    )    

    val_ds = SegmentationDataset(
        transforms=get_val_seg_transforms,
        split='val',
        img_resize=IMG_RESIZE
    )

    # Create Dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    val_dl = DataLoader(
        val_ds, 
        batch_size=2,
        shuffle=True,
        num_workers=2
    )

    seg_trainer = SegTrainer(
        model_run_name=MODEL_RUN_NAME,
        model=model,
        device=device,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR
    )

    # logger.info(f"Created model trainer class {type(seg_trainer)}")

    # seg_trainer.train()
    model.eval()

    state_dict = torch.load('checkpoints/segmentation_models/Segformer_mit_b1_1024_epoch_10_vloss-0.032614.pth', map_location=device)['model_state_dict']
    model.load_state_dict(state_dict)

    for index in range(len(val_ds)):
        img, target, img_id = val_ds[index]
        print(img.shape, target.shape, img_id)

        img = img.unsqueeze(0)
        logits = model(img)

        # upsampled_logits = F.interpolate(output['logits'], size=(512, 512), mode="bilinear", align_corners=False)
        preds = argmax(logits, dim=1).squeeze(0).cpu().numpy()

        print(preds.shape)

        cv2.imwrite(Path('outputs/segmentation') / (img_id + "_preds.png"), preds*50)

if __name__ == '__main__':
    main()