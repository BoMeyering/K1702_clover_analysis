"""
train_object_detection_model.py
Main Training Script for the Object Detection Model (DDP version)
BoMeyering 2025
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
import logging

from src.models import create_fasterrcnn_model
from src.datasets import ObjDetDataset
from src.transforms import get_train_obj_transforms, get_val_obj_transforms
from src.trainer import ObjTrainer
from src.utils.loggers import setup_loggers

# CONFIG
model_config = {
    "architecture": "fasterrcnn_resnet50_fpn",
    "pretrained": True,
    "num_classes": 2,   # 1 class + background
    "max_det_per_image": 20,
    "image_size": (1024, 1024)
}

EPOCHS = 10
CHECKPOINT_DIR = 'checkpoints/object_detection_models'
MODEL_RUN_NAME = "_".join([model_config['architecture'], str(model_config['image_size'][0])])

# Logging (only on rank 0)
def setup_logger(rank):
    if rank == 0:
        setup_loggers(model_run=MODEL_RUN_NAME, log_dir='logs', log_level='INFO')
    logger = logging.getLogger()
    return logger

# TRAINING FUNCTION
def main(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    logger = setup_logger(rank)
    if rank == 0:
        logger.info(f"Running DDP training on rank {rank}/{world_size} with GPU {rank}")

    # Model
    model = create_fasterrcnn_model(**model_config).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    # Optimizer + Scheduler
    optimizer = SGD(params=model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)

    # Datasets + Dataloaders
    train_transforms = get_train_obj_transforms(resize=model_config['image_size'])
    val_transforms = get_val_obj_transforms(resize=model_config['image_size'])

    train_ds = ObjDetDataset(transforms=train_transforms)
    val_ds = ObjDetDataset(transforms=val_transforms, split='val')

    train_dl = train_ds.get_dataloader(
        batch_size=4,
        distributed=True,
        num_replicas=world_size,
        rank=rank
    )
    val_dl = val_ds.get_dataloader(
        batch_size=4,
        distributed=True,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Trainer
    objdet_trainer = ObjTrainer(
        model_run_name=MODEL_RUN_NAME,
        model=model,
        device=rank,
        train_loader=train_dl,
        val_loader=val_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
        is_master=(rank == 0)
    )

    if rank == 0:
        logger.info(f"Created model trainer class {type(objdet_trainer)}")

    objdet_trainer.train()

    dist.destroy_process_group()

# ENTRY POINT
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    rank = int(os.environ["RANK"])
    main(rank, world_size)
