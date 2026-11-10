"""
src/ssl/distributed.py
Torch distributed functions
Adapted from pgcview_v2_training_framework/src/distributed.py: also runs as a single process without torchrun.
BoMeyering 2025
"""

import os

import torch.distributed as dist
from omegaconf import OmegaConf


def setup_ddp(backend: str):
    """Set up DDP. Without torchrun, initialise a one-process group so the same code path works."""
    if "RANK" not in os.environ:
        os.environ.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"})
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29531")
    dist.init_process_group(backend=backend, init_method="env://")


def shutdown_ddp():
    """Shutdown the distributed processes"""
    dist.barrier()
    dist.destroy_process_group()


def set_env_ranks(conf: OmegaConf):
    """Get ranks and world size for a process"""
    conf.rank = int(os.environ["RANK"])
    conf.local_rank = int(os.environ["LOCAL_RANK"])
    conf.world_size = int(os.environ["WORLD_SIZE"])
    conf.is_main = is_main_process()


def is_main_process() -> bool:
    """Returns True if process is the main process"""
    return (
        (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    )
