"""
src.ssl.trainer.py
FlexMatch trainer for semantic segmentation with a tiny labeled set.
Ported from pgcview_v2_training_framework/src/trainer.py (FlexMatchTrainer), with these changes:

  * weak/strong views come pixel-aligned from the dataloader (no CPU round trip, no double normalisation);
    CutMix is applied on the GPU to the strong view, pseudo-labels and confidence mask;
  * stateful FlexMatchThreshold (running, DDP-synchronised learning status) replaces per-batch class_beta;
  * bf16 autocast; samplers get set_epoch; optional supervised warmup and a supervised-only baseline mode;
  * evaluation on held-out HQ images (EMA weights, logits upsampled to decoded resolution, ignore pixels
    excluded) and an optional proxy evaluation against the old SAM masks on unlabeled images;
  * the EMA state dict is cloned inside apply_ema (previously it aliased the live weights, so the saved
    "EMA" checkpoint was really the raw model).

BoMeyering 2025
"""

import json
import logging
import time
import uuid
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchmetrics import MeanMetric
from tqdm import tqdm

from src.ssl.callbacks import CheckpointManager
from src.ssl.datasets import IGNORE_INDEX
from src.ssl.distributed import is_main_process
from src.ssl.flexmatch import FlexMatchThreshold, apply_cutmix, cutmix_boxes
from src.ssl.loggers import rank_log
from src.ssl.metrics import MetricLogger
from src.ssl.parameters import EMA, apply_ema

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None


def update_metrics(metric_logger: MetricLogger, preds: torch.Tensor, targets: torch.Tensor):
    """Update a MetricLogger, dropping ignore pixels (each image is flattened to its valid pixels)."""
    valid = targets != IGNORE_INDEX
    if bool(valid.all()):
        metric_logger.update(preds=preds, targets=targets)
        return
    for p, t, v in zip(preds, targets, valid):
        if v.any():
            metric_logger.update(preds=p[v].unsqueeze(0), targets=t[v].unsqueeze(0))


def to_float(x) -> Optional[float]:
    if x is None:
        return None
    return x.item() if isinstance(x, torch.Tensor) else float(x)


class FlexMatchSegTrainer:
    """FlexMatch (or supervised-only) training loop for one labeled and one unlabeled pool."""

    def __init__(
        self,
        conf: OmegaConf,
        model: torch.nn.Module,
        l_loader,
        u_loader,
        samplers: List,
        steps_per_epoch: int,
        val_loader,
        proxy_loader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        checkpoint_manager: CheckpointManager,
        ema: Optional[EMA] = None,
    ):
        self.conf = conf
        self.model = model                  # DDP-wrapped, used for training
        self.eval_model = model.module      # unwrapped, used for eval (no DDP collectives on single-rank calls)
        self.l_loader, self.u_loader = l_loader, u_loader
        self.samplers = samplers
        self.steps_per_epoch = steps_per_epoch
        self.val_loader, self.proxy_loader = val_loader, proxy_loader
        self.criterion = criterion
        self.optimizer, self.scheduler = optimizer, scheduler
        self.checkpoint_manager = checkpoint_manager
        self.ema = ema
        self.device = conf.device
        self.fm = conf.flexmatch
        self.logger = logging.getLogger()
        self.trainer_id = f'flexmatch_trainer_{uuid.uuid4()}'
        self.amp = bool(conf.training.amp) and 'cuda' in self.device

        with open(conf.metadata.target_mapping_path) as f:
            mapping = json.load(f)
        self.class_names = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1]['class_idx'])]
        self.num_classes = conf.model.config.classes
        self.palette = np.zeros((256, 3), dtype=np.uint8)
        for v in mapping.values():
            self.palette[v['class_idx']] = v['rgb']

        # HQ class index -> SAM class index, for proxy evaluation and pseudo-label diagnostics
        proxy = conf.data.proxy
        self.sam_classes = list(proxy.sam_classes)
        hq_to_sam = torch.full((256,), IGNORE_INDEX, dtype=torch.long)
        for name, v in mapping.items():
            hq_to_sam[v['class_idx']] = proxy.class_map[name]
        self.hq_to_sam = hq_to_sam.to(self.device)

        self.thresholder = FlexMatchThreshold(
            num_classes=self.num_classes, tau=self.fm.tau, mapping=self.fm.mapping, warmup=self.fm.warmup,
            status=self.fm.status, normalize=self.fm.normalize, momentum=self.fm.momentum,
            tau_min=self.fm.tau_min, device=self.device,
        )

        # Meters (MeanMetric syncs across ranks on compute)
        meter_names = ['loss', 'l_loss', 'u_loss', 'conf_frac', 'pl_acc_conf', 'pl_acc_all', 'val_loss']
        self.meters = {k: MeanMetric().to(self.device) for k in meter_names}

        self.train_metrics = MetricLogger('Train Metrics (labeled)', self.num_classes, self.device)
        self.val_metrics = MetricLogger('HQ Holdout Metrics', self.num_classes, self.device)
        self.proxy_metrics = MetricLogger('Proxy Metrics (vs SAM masks)', len(self.sam_classes), self.device)

        self.run = None
        if conf.is_main and conf.wandb.enabled and conf.wandb.mode != 'disabled':
            if wandb is None:
                rank_log(True, self.logger.warning, 'wandb is not installed; continuing without it.')
            else:
                self.run = wandb.init(
                    project=conf.wandb.project, entity=conf.wandb.entity, name=conf.model_run,
                    mode=conf.wandb.mode, config=OmegaConf.to_container(conf, resolve=True),
                )

    # ------------------------------------------------------------------ helpers
    def _autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp)

    def _ssl_active(self, epoch: int) -> bool:
        return bool(self.fm.enabled) and epoch > self.fm.unsup_warmup_epochs

    def _labeled_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets != IGNORE_INDEX
        if bool(valid.all()):
            return self.criterion(logits, targets)
        return self.criterion(logits, torch.where(valid, targets, torch.zeros_like(targets)), valid)

    # ------------------------------------------------------------------ training
    def _train_step(self, l_batch, u_batch, ssl: bool) -> Dict[str, torch.Tensor]:
        l_img, l_targets, _ = l_batch
        l_img = l_img.to(self.device, non_blocking=True)
        l_targets = l_targets.to(self.device, non_blocking=True)
        n_l = l_img.shape[0]
        out = {}

        if not ssl:
            with self._autocast():
                l_logits = self.model(l_img)
            l_loss = self._labeled_loss(l_logits.float(), l_targets)
            out.update(loss=l_loss, l_loss=l_loss.detach(), u_loss=torch.zeros((), device=self.device))
        else:
            weak, strong, sam, _ = u_batch
            weak = weak.to(self.device, non_blocking=True)
            strong = strong.to(self.device, non_blocking=True)
            sam = sam.to(self.device, non_blocking=True)

            # Pseudo-label the weak view with the live model (FixMatch/FlexMatch convention)
            with torch.no_grad(), self._autocast():
                weak_logits = self.model(weak)
            probs = torch.softmax(weak_logits.float(), dim=1)
            pseudo, conf_mask = self.thresholder(probs)

            # Diagnostics: agreement of (collapsed) pseudo-labels with the old SAM masks
            sam_valid = sam != IGNORE_INDEX
            if sam_valid.any():
                agree = (self.hq_to_sam[pseudo] == sam)
                out['pl_acc_all'] = agree[sam_valid].float().mean()
                cm = sam_valid & conf_mask
                if cm.any():
                    out['pl_acc_conf'] = agree[cm].float().mean()

            # CutMix on the strong view, pseudo-labels and mask together
            if self.fm.cutmix_p > 0:
                boxes = cutmix_boxes(strong.shape[0], strong.shape[2], strong.shape[3], self.fm.cutmix_p, self.device)
                strong, pseudo, conf_mask = apply_cutmix(strong, pseudo, conf_mask, boxes)

            with self._autocast():
                logits = self.model(torch.cat([l_img, strong]))
            l_logits, s_logits = logits[:n_l].float(), logits[n_l:].float()

            l_loss = self._labeled_loss(l_logits, l_targets)
            f = conf_mask.float().mean()
            if bool(conf_mask.any()):
                u_loss = self.criterion(s_logits, pseudo, conf_mask)
                if self.fm.scale_by_conf_fraction:
                    u_loss = u_loss * f
            else:
                u_loss = s_logits.sum() * 0.0   # keep the graph identical across ranks
            u_loss = self.fm.lam * u_loss
            out.update(loss=l_loss + u_loss, l_loss=l_loss.detach(), u_loss=u_loss.detach(), conf_frac=f)

        update_metrics(self.train_metrics, l_logits.detach().argmax(1), l_targets)
        return out

    def _train_epoch(self, epoch: int):
        self.model.train()
        for m in self.meters.values():
            m.reset()
        self.train_metrics.reset()
        for s in self.samplers:
            if hasattr(s, 'set_epoch'):
                s.set_epoch(epoch)

        ssl = self._ssl_active(epoch)
        l_iter: Iterator = iter(self.l_loader)
        u_iter: Optional[Iterator] = iter(self.u_loader) if (ssl and self.u_loader is not None) else None

        p_bar = tqdm(range(self.steps_per_epoch), colour='yellow', disable=not is_main_process())
        for step in p_bar:
            self.optimizer.zero_grad(set_to_none=True)
            l_batch = next(l_iter)
            u_batch = next(u_iter) if u_iter is not None else None
            out = self._train_step(l_batch, u_batch, ssl)
            out['loss'].backward()
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update_params()

            for k in ('loss', 'l_loss', 'u_loss', 'conf_frac', 'pl_acc_conf', 'pl_acc_all'):
                if k in out:
                    self.meters[k].update(out[k].detach())
            p_bar.set_description(
                f"Train {epoch}/{self.conf.training.epochs} {'SSL' if ssl else 'SUP'}. "
                f"LR {self.scheduler.get_last_lr()[0]:.2e}. Loss {out['loss'].item():.4f} "
                f"(L {out['l_loss'].item():.4f} U {out['u_loss'].item():.4f}). "
                f"Conf {to_float(out.get('conf_frac', 0.0)):.3f}"
            )

        dist.barrier()
        # Every rank computes the same meters in the same order (compute() is a collective under DDP)
        keys = ('loss', 'l_loss', 'u_loss') + (('conf_frac', 'pl_acc_conf', 'pl_acc_all') if ssl else ())
        logs = {}
        for k in keys:
            v = to_float(self.meters[k].compute())   # nan if no rank updated it
            if v is not None and not np.isnan(v):
                logs[k] = v
        self.train_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.train_metrics)
        return logs

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def _predict_full(self, img: torch.Tensor, size) -> torch.Tensor:
        with self._autocast():
            logits = self.eval_model(img.to(self.device, non_blocking=True))
        return F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Optional[float]:
        if self.val_loader is None:
            return None
        self.eval_model.eval()
        self.meters['val_loss'].reset()
        self.val_metrics.reset()
        with apply_ema(self.ema):
            for img, target, _ in tqdm(self.val_loader, colour='blue', disable=not is_main_process(), desc=f'HQ val {epoch}'):
                target = target.to(self.device)
                logits = self._predict_full(img, target.shape[-2:])
                self.meters['val_loss'].update(self._labeled_loss(logits, target))
                update_metrics(self.val_metrics, logits.argmax(1), target)
        dist.barrier()
        val_loss = to_float(self.meters['val_loss'].compute())
        self.val_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metrics)
        return val_loss

    @torch.no_grad()
    def _proxy_epoch(self, epoch: int) -> bool:
        if self.proxy_loader is None:
            return False
        self.eval_model.eval()
        self.proxy_metrics.reset()
        with apply_ema(self.ema):
            for img, target, _ in tqdm(self.proxy_loader, colour='cyan', disable=not is_main_process(), desc=f'Proxy {epoch}'):
                target = target.to(self.device)
                preds = self._predict_full(img, target.shape[-2:]).argmax(1)
                update_metrics(self.proxy_metrics, self.hq_to_sam[preds], target)
        dist.barrier()
        self.proxy_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.proxy_metrics)
        return True

    @torch.no_grad()
    def _vis_images(self) -> list:
        """Image | ground truth | prediction panels for the HQ holdout (EMA weights)."""
        if self.run is None or self.val_loader is None:
            return []
        means = np.array(self.conf.metadata.norm.means)[:, None, None]
        stds = np.array(self.conf.metadata.norm.std)[:, None, None]
        alpha = self.conf.wandb.vis_alpha
        panels = []
        dataset = self.val_loader.dataset
        self.eval_model.eval()
        with apply_ema(self.ema):
            for idx in range(min(len(dataset), self.conf.wandb.num_vis_images)):
                img, target, img_ids = dataset[idx]
                pred = self._predict_full(img.unsqueeze(0), target.shape[-2:]).argmax(1)[0].cpu().numpy()
                raw = np.clip((img.numpy() * stds + means) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)
                h, w = target.shape
                raw = np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(raw).permute(2, 0, 1)[None].float(), size=(h, w), mode='bilinear'
                )[0].permute(1, 2, 0).clamp(0, 255).byte())
                gt = self.palette[target.numpy().clip(0, 255)]
                pr = self.palette[pred]
                blend = lambda c: np.clip(alpha * c + (1 - alpha) * raw, 0, 255).astype(np.uint8)
                panel = np.concatenate([raw, blend(gt), blend(pr)], axis=1)[::4, ::4]
                panels.append(wandb.Image(panel, caption=f'{img_ids}: image | HQ label | prediction'))
        return panels

    # ------------------------------------------------------------------ main loop
    def _metric_dict(self, metric_logger: MetricLogger, names: List[str], prefix: str) -> dict:
        res = metric_logger.results
        if not res or res.get('avg') is None:
            return {}
        d = {f'{prefix}/{k}': to_float(v) for k, v in res['avg'].items()}
        for k, v in (res.get('mc') or {}).items():
            for i, name in enumerate(names):
                if i < v.numel():
                    d[f'{prefix}/{k}/{name}'] = to_float(v[i])
        return d

    def train(self):
        rank_log(self.conf.is_main, self.logger.info,
                 f'Training {self.trainer_id} for {self.conf.training.epochs} epochs x {self.steps_per_epoch} steps.')
        proxy_every = max(1, int(self.conf.data.proxy.every_n_epochs))

        for epoch in range(1, self.conf.training.epochs + 1):
            t0 = time.time()
            train_logs = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)
            ran_proxy = (epoch % proxy_every == 0 or epoch == self.conf.training.epochs) and self._proxy_epoch(epoch)

            thresholds = self.thresholder.thresholds().cpu()
            status = self.thresholder.learning_status().cpu()
            rank_log(self.conf.is_main, self.logger.info,
                     f"Epoch {epoch} ({time.time() - t0:.0f}s) - train {train_logs} - val_loss {val_loss} - "
                     f"thresholds {dict(zip(self.class_names, [round(t, 3) for t in thresholds.tolist()]))}")

            val = self._metric_dict(self.val_metrics, self.class_names, 'val') if val_loss is not None else {}
            if self.run is not None:
                log = {'epoch': epoch, 'lr': self.scheduler.get_last_lr()[0], 'val/loss': val_loss}
                log.update({f'train/{k}': v for k, v in train_logs.items()})
                log.update(val)
                log.update(self._metric_dict(self.train_metrics, self.class_names, 'train'))
                if ran_proxy:
                    log.update(self._metric_dict(self.proxy_metrics, self.sam_classes, 'proxy'))
                if self._ssl_active(epoch):
                    for i, name in enumerate(self.class_names):
                        log[f'flexmatch/threshold/{name}'] = thresholds[i].item()
                        log[f'flexmatch/status/{name}'] = status[i].item()
                    log['flexmatch/unused_frac'] = self.thresholder.unused_frac.item()
                log['val/predictions'] = self._vis_images()
                self.run.log(log)
            dist.barrier()

            # Checkpoint (EMA weights cloned inside apply_ema so they are not overwritten on restore)
            with apply_ema(self.ema):
                ema_state = {k: v.detach().cpu().clone() for k, v in self.model.module.state_dict().items()}
            if val_loss is not None:
                score = dict(val_loss=val_loss, MeanIoU=val.get('val/MeanIoU'), GeneralizedDiceScore=val.get('val/GeneralizedDiceScore'))
            elif ran_proxy or self.proxy_loader is None:
                pm = self._metric_dict(self.proxy_metrics, self.sam_classes, 'proxy')
                score = dict(val_loss=0.0, MeanIoU=pm.get('proxy/MeanIoU'), GeneralizedDiceScore=pm.get('proxy/GeneralizedDiceScore'))
            else:
                score = None
            if score is not None and self.conf.is_main:
                self.checkpoint_manager(logs={
                    'epoch': epoch, **score,
                    'model_state_dict': {k: v.detach().cpu().clone() for k, v in self.model.module.state_dict().items()},
                    'ema_state_dict': ema_state,
                    'flexmatch_state': self.thresholder.state_dict(),
                    'class_names': self.class_names,
                    'config': OmegaConf.to_container(self.conf, resolve=True),
                })
            dist.barrier()
            if self.scheduler:
                self.scheduler.step()

        if self.run is not None:
            self.run.finish()
        rank_log(self.conf.is_main, self.logger.info, 'Training complete')
