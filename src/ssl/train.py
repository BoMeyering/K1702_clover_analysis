"""
src/ssl/train.py
FlexMatch / supervised segmentation training on the HQ Labelbox masks: the HQ images are the labeled
set, every other image is the unlabeled set. Adapted from pgcview_v2_training_framework/train_flexmatch.py.

Entry points (all equivalent; `k1702 train` launches torchrun for you on multi-GPU machines):
    uv run k1702 train -c configs/flexmatch_sainfoin.yaml                 # all visible GPUs
    uv run k1702 train -c configs/flexmatch_sainfoin.yaml --gpus 1 training.epochs=1 wandb.mode=disabled
    uv run torchrun --nproc_per_node=2 train_flexmatch.py -c configs/flexmatch_sainfoin.yaml

Any config value can be overridden with dotted key=value arguments.
BoMeyering 2025
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import segmentation_models_pytorch as smp
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.ssl.callbacks import CheckpointManager
from src.ssl.config import TrainSSLConfig, set_run_name
from src.ssl.dataloaders import DistributedInfiniteSampler
from src.ssl.datasets import LabeledDataset, UnlabeledDataset
from src.ssl.distributed import set_env_ranks, setup_ddp, shutdown_ddp
from src.ssl.loggers import rank_log, setup_loggers
from src.ssl.losses import get_loss_criterion, read_class_counts
from src.ssl.parameters import EMA, OptimConfig
from src.ssl.trainer import FlexMatchSegTrainer
from src.ssl.transforms import (get_geometric_transforms, get_labeled_photometric_transforms, get_normalize_transforms,
                                get_strong_photometric_transforms, get_val_transforms)


def build_conf(config_path: str, overrides: Sequence[str] = ()) -> OmegaConf:
    """Merge the structured defaults, the YAML config and dotted overrides, then fill in rank / device / class info."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file {config_path} not found.')
    conf = OmegaConf.merge(OmegaConf.structured(TrainSSLConfig), OmegaConf.load(config_path), OmegaConf.from_dotlist(list(overrides)))
    set_env_ranks(conf)
    set_run_name(conf)

    if 'cuda' in conf.device and torch.cuda.is_available():
        conf.device = f'cuda:{conf.local_rank}'
        torch.cuda.set_device(conf.local_rank)
    else:
        conf.device = 'cpu'

    with open(conf.metadata.target_mapping_path) as f:
        target_mapping = json.load(f)
    n_classes = len({v['class_idx'] for v in target_mapping.values()})
    if conf.model.config.classes is None:
        conf.model.config.classes = n_classes
    elif conf.model.config.classes != n_classes:
        raise ValueError(f'model.config.classes={conf.model.config.classes} but the target mapping has {n_classes} classes')

    if conf.loss.get('class_sample_count_path', None):
        conf.loss.samples, conf.loss.weights = read_class_counts(conf.loss.class_sample_count_path)
    else:
        del conf.loss.samples
        del conf.loss.weights
    return conf


def build_id_pools(conf):
    """Labeled / holdout / unlabeled / proxy-eval image id lists."""
    d = conf.data
    exclude = set(d.exclude_ids)
    hq_ids = sorted(p.stem for p in Path(d.hq_target_dir).glob('*.png'))
    holdout = sorted(set(d.holdout_ids))
    if not set(holdout) <= set(hq_ids):
        raise ValueError(f'holdout_ids not in {d.hq_target_dir}: {sorted(set(holdout) - set(hq_ids))}')
    labeled = sorted(set(hq_ids) - set(holdout) - exclude)

    if d.split_csv:
        split = pl.read_csv(d.split_csv)
        all_ids = {i for i in split['img_id'].to_list() if (Path(d.image_dir) / f'{i}.jpg').exists()}
    else:
        if d.proxy.enabled:
            raise ValueError('data.proxy.enabled needs data.split_csv to choose the proxy images')
        all_ids = {p.stem for p in Path(d.image_dir).glob('*.jpg')}
    missing_hq = set(hq_ids) - all_ids
    if missing_hq:
        raise FileNotFoundError(f'HQ targets without an image in {d.image_dir}: {sorted(missing_hq)}')
    proxy = []
    if d.proxy.enabled:
        proxy = sorted(set(split.filter(pl.col('split') == d.proxy.split)['img_id'].to_list()) & all_ids - set(hq_ids) - exclude)
        if d.proxy.max_images:
            proxy = proxy[: d.proxy.max_images]
    unlabeled = all_ids - set(hq_ids) - exclude   # all HQ images (holdout included) stay out of the unlabeled pool
    if d.unlabeled_exclude_proxy:
        unlabeled -= set(proxy)
    return labeled, holdout, sorted(unlabeled), proxy


def eval_loader(ds, conf):
    if ds is None or len(ds) == 0:
        return None
    sampler = DistributedSampler(ds, num_replicas=conf.world_size, rank=conf.rank, shuffle=False, drop_last=False)
    return DataLoader(ds, batch_size=1, sampler=sampler, num_workers=conf.training.val_num_workers, pin_memory=True,
                      persistent_workers=conf.training.val_num_workers > 0)


def train(conf):
    logger = logging.getLogger()
    rank_log(conf.is_main, logger.info, 'Current Training Configuration\n' + OmegaConf.to_yaml(conf))
    labeled, holdout, unlabeled, proxy = build_id_pools(conf)
    rank_log(conf.is_main, logger.info,
             f'Labeled: {len(labeled)} {labeled}\nHoldout: {len(holdout)} {holdout}\n'
             f'Unlabeled: {len(unlabeled)}  Proxy eval ({conf.data.proxy.split}): {len(proxy)}  World size: {conf.world_size}')

    # Model
    model = getattr(smp, conf.model.architecture.value)(**OmegaConf.to_container(conf.model.config)).to(conf.device)
    model = DDP(model, device_ids=[conf.local_rank] if 'cuda' in conf.device else None,
                find_unused_parameters=conf.training.find_unused_parameters)
    rank_log(conf.is_main, logger.info, f'Created {conf.model.architecture.value} / {conf.model.config.encoder_name} with {conf.model.config.classes} classes')

    # Data
    resize, red = tuple(conf.images.resize), conf.images.decode_reduction
    norm = get_normalize_transforms(conf.metadata.norm.means, conf.metadata.norm.std)
    geometric = get_geometric_transforms(resize, scale=conf.images.crop_scale, ratio=conf.images.crop_ratio)
    l_ds = LabeledDataset(labeled, conf.data.image_dir, conf.data.hq_target_dir, geometric=geometric,
                          photometric=get_labeled_photometric_transforms(), normalize=norm, reduction=red)
    l_sampler = DistributedInfiniteSampler(l_ds, num_replicas=conf.world_size, rank=conf.rank, shuffle=True, seed=conf.training.seed)
    l_loader = DataLoader(l_ds, batch_size=conf.flexmatch.l_batch_size, sampler=l_sampler, drop_last=True,
                          num_workers=conf.training.l_num_workers, pin_memory=True, persistent_workers=conf.training.l_num_workers > 0)
    samplers = [l_sampler]

    per_rank_unlabeled = len(unlabeled) // conf.world_size
    steps_per_epoch = conf.training.steps_per_epoch or max(1, per_rank_unlabeled // conf.flexmatch.u_batch_size)

    u_loader = None
    if conf.flexmatch.enabled:
        u_ds = UnlabeledDataset(unlabeled, conf.data.image_dir, geometric, get_strong_photometric_transforms(), norm, reduction=red,
                                sam_target_dir=conf.data.proxy.target_dir if (conf.data.sam_diagnostics and conf.data.proxy.enabled) else None)
        u_sampler = DistributedInfiniteSampler(u_ds, num_replicas=conf.world_size, rank=conf.rank, shuffle=True, seed=conf.training.seed)
        u_loader = DataLoader(u_ds, batch_size=conf.flexmatch.u_batch_size, sampler=u_sampler, drop_last=True,
                              num_workers=conf.training.u_num_workers, pin_memory=True, persistent_workers=conf.training.u_num_workers > 0)
        samplers.append(u_sampler)

    val_tf = get_val_transforms(resize, conf.metadata.norm.means, conf.metadata.norm.std)
    val_ds = LabeledDataset(holdout, conf.data.image_dir, conf.data.hq_target_dir, image_only=val_tf, reduction=red) if holdout else None
    proxy_ds = LabeledDataset(proxy, conf.data.image_dir, conf.data.proxy.target_dir, image_only=val_tf, reduction=red) if proxy else None

    # Optimizer, scheduler, loss, EMA
    model, optimizer, scheduler = OptimConfig(conf=conf, model=model).process()
    criterion = get_loss_criterion(conf)
    ema = EMA(model, decay=conf.optimizer.ema_decay) if conf.optimizer.ema else None
    rank_log(conf.is_main, logger.info, f'Loss {type(criterion).__name__}; EMA {"on, decay " + str(conf.optimizer.ema_decay) if ema else "off"}')

    if conf.directories.starting_checkpoint_path:
        chkpt = torch.load(conf.directories.starting_checkpoint_path, map_location='cpu', weights_only=False)
        state = {k.removeprefix('module.'): v for k, v in chkpt.get('model_state_dict', chkpt).items()}
        model.module.load_state_dict(state)
        rank_log(conf.is_main, logger.info, f'Loaded starting weights from {conf.directories.starting_checkpoint_path}')

    checkpoint_manager = CheckpointManager(conf=conf, top_k=conf.checkpoint.top_k, w_loss=conf.checkpoint.w_loss,
                                           w_miou=conf.checkpoint.w_miou, w_gds=conf.checkpoint.w_gds)

    trainer = FlexMatchSegTrainer(
        conf=conf, model=model, l_loader=l_loader, u_loader=u_loader, samplers=samplers,
        steps_per_epoch=steps_per_epoch, val_loader=eval_loader(val_ds, conf), proxy_loader=eval_loader(proxy_ds, conf),
        criterion=criterion, optimizer=optimizer, scheduler=scheduler, checkpoint_manager=checkpoint_manager, ema=ema,
    )
    trainer.train()


def run(config_path: str, overrides: Sequence[str] = (), backend: str = 'nccl'):
    """Train in this process: one rank under torchrun, or a one-process group without it."""
    setup_ddp(backend=backend)
    conf = build_conf(config_path, overrides)
    Path(conf.directories.log_dir).mkdir(parents=True, exist_ok=True)
    if conf.is_main:
        setup_loggers(conf)

    # Different augmentation streams per rank, reproducible per run
    seed = conf.training.seed + conf.rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train(conf)
    shutdown_ddp()   # not in a finally: its barrier would hang the other ranks after a crash
