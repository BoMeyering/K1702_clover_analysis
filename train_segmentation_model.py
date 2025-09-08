"""
train_segmentation_model.py
Main Training Script for the Segmentation Model (DDP version)
BoMeyering 2025
"""

import os
import torch
import torch.distributed as dist
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss

from src.models import create_smp_model
from src.datasets import SegmentationDataset
from src.transforms import get_train_seg_transforms, get_val_seg_transforms
from src.trainer import SegTrainer
from src.utils.loggers import setup_loggers

# CONFIG
model_config = {
    "architecture": "Segformer",
    "encoder_name": "mit_b1",
    "encoder_depth": 5,
    "encoder_weights": "imagenet",
    "input_channels": 3,
    "classes": 3
}

EPOCHS = 10
CHECKPOINT_DIR = "checkpoints/segmentation_models"
IMG_RESIZE = (1024, 1024)
MODEL_RUN_NAME = "_".join([model_config["architecture"], model_config["encoder_name"], str(IMG_RESIZE[0])])

# Logging helper (only creates loggers on rank 0)
def setup_logger(rank):
    if rank == 0:
        setup_loggers(model_run=MODEL_RUN_NAME, log_dir="logs", log_level="INFO")
    return logging.getLogger()


def make_dataloader_from_dataset(ds, batch_size, num_workers, distributed, world_size, rank, shuffle=True):
    """Helper: if dataset has get_dataloader use it, otherwise fall back to DistributedSampler + DataLoader"""
    if hasattr(ds, "get_dataloader") and callable(getattr(ds, "get_dataloader")):
        # follow the pattern used for object detection dataset API
        return ds.get_dataloader(
            batch_size=batch_size,
            distributed=distributed,
            num_replicas=world_size,
            rank=rank,
            num_workers=num_workers,
            shuffle=shuffle
        )
    else:
        if distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle)
        else:
            sampler = None
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None and shuffle),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )


def main(rank, world_size):
    # init process group (torchrun sets env for init_method="env://")
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    logger = setup_logger(rank)
    if rank == 0:
        logger.info(f"Running DDP segmentation training on rank {rank}/{world_size} with GPU {rank}")

    # device (we pass rank to trainer to match object detection style)
    device = rank

    # create model and move to device
    model = create_smp_model(config=model_config)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    if rank == 0:
        logger.info(f"Instantiated segmentation model {type(model)} and sent to device {device}.")

    # optimizer, scheduler, criterion
    optimizer = SGD(params=model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
    criterion = CrossEntropyLoss()

    # transforms
    train_transforms = get_train_seg_transforms(resize=IMG_RESIZE)
    val_transforms = get_val_seg_transforms(resize=IMG_RESIZE)

    # datasets
    train_ds = SegmentationDataset(transforms=train_transforms)
    val_ds = SegmentationDataset(transforms=val_transforms, split="val")

    # dataloaders (use dataset's get_dataloader if available to match detection code)
    train_dl = make_dataloader_from_dataset(
        train_ds,
        batch_size=2,
        num_workers=2,
        distributed=True,
        world_size=world_size,
        rank=rank,
        shuffle=True
    )

    val_dl = make_dataloader_from_dataset(
        val_ds,
        batch_size=2,
        num_workers=2,
        distributed=True,
        world_size=world_size,
        rank=rank,
        shuffle=False
    )

    # trainer
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
        checkpoint_dir=CHECKPOINT_DIR,
        # SegTrainer will read is_master from BaseTrainer, but keep explicit param for parity
        # (SegTrainer signature accepts use_amp; no is_master param needed -- trainer reads rank)
    )

    if rank == 0:
        logger.info(f"Created segmentation trainer class {type(seg_trainer)}")

    try:
        seg_trainer.train()
    finally:
        # ensure process group cleanup
        dist.destroy_process_group()


if __name__ == "__main__":
    # When using torchrun, RANK and WORLD_SIZE are set by launcher.
    world_size = torch.cuda.device_count()
    # If user runs with torchrun, the spawn is handled externally; torchrun will set RANK for each process.
    # So we spawn the main per-process by reading RANK from env (match your detection script behavior).
    rank = int(os.environ.get("RANK", 0))
    # ensure MASTER_ADDR/MASTER_PORT are set by the environment when using torchrun; otherwise use defaults
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    main(rank, world_size)
