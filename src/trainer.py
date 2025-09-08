"""
src.trainer.py
Trainer Classes
BoMeyering 2025
"""
import os
import logging
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from abc import ABC, abstractmethod
from src.eval import AverageMeter, AverageMeterSet
from src.callbacks import ModelCheckpoint
from src.metrics import ObjectDetectionMetricLogger, SegmentationMetricLogger
import torch.distributed as dist


class BaseTrainer(ABC):
    def __init__(
        self,
        model_run_name,
        model,
        device,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs,
        checkpoint_dir
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.logger = logging.getLogger()
        self.checkpoint = ModelCheckpoint(checkpoint_dir=checkpoint_dir, model_run_name=model_run_name)
        self.meters = AverageMeterSet()

        # Distributed rank/world flags & master check
        if dist.is_available() and dist.is_initialized():
            try:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
            except Exception:
                self.rank = int(os.environ.get("RANK", 0))
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        else:
            self.rank = int(os.environ.get("RANK", 0))
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_master = (self.rank == 0)

    def train(self):
        """ Main train method """
        self.logger.info(f"Training model for {self.epochs} epochs.")
        for epoch in range(1, self.epochs + 1):
            # If DistributedSampler is used, set epoch for shuffling
            try:
                if hasattr(self.train_loader, "sampler") and isinstance(self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    self.train_loader.sampler.set_epoch(epoch)
            except Exception:
                pass

            self._train_epoch(epoch)
            self._val_epoch(epoch)

            logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(self.meters['train_loss'].avg, device=self.device),
                "val_loss": torch.tensor(self.meters['val_loss'].avg, device=self.device),
                "model_state_dict": self.model.state_dict(),
            }

            self.logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {self.meters['train_loss'].avg:.6f}, "
                f"Val Loss: {self.meters['val_loss'].avg:.6f}"
            )

            if dist.is_initialized():
                dist.barrier()
            if self.is_master:
                self.checkpoint(epoch=epoch, logs=logs)

            if self.scheduler:
                self.scheduler.step()

        self.logger.info("Training complete")

    @abstractmethod
    def _train_epoch(self, epoch):
        pass

    @abstractmethod
    def _train_step(self, batch):
        pass

    @abstractmethod
    def _val_epoch(self, epoch):
        pass

    @abstractmethod
    def _val_step(self, batch):
        pass


class SegTrainer(BaseTrainer):
    def __init__(self, *args, criterion, num_classes=3, use_amp=None, **kwargs):
        """
        Segmentation trainer ready for multi-GPU DDP.

        Args:
            num_classes (int): number of classes in segmentation task
            use_amp (bool|None): whether to use automatic mixed precision. If None, AMP will be enabled when using CUDA.
        """
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.seg_metrics = SegmentationMetricLogger(num_classes=num_classes, device=self.device)

        if use_amp is None:
            self.use_amp = torch.cuda.is_available() and ("cuda" in str(self.device).lower())
        else:
            self.use_amp = bool(use_amp)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _train_epoch(self, epoch):
        self.model.train()
        self.meters.reset()

        if self.is_master:
            self.logger.info(f"Training epoch {epoch}")
            p_bar = tqdm(self.train_loader)
        else:
            p_bar = self.train_loader

        for batch_idx, batch in enumerate(p_bar):
            self.model.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    train_loss = self._train_step(batch)
                self.scaler.scale(train_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                train_loss = self._train_step(batch)
                train_loss.backward()
                self.optimizer.step()

            try:
                loss_value = train_loss.item()
            except Exception:
                loss_value = float(train_loss)
            self.meters.update('train_loss', loss_value)

            if self.is_master:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                p_bar.set_description(
                    f"Train Epoch: {epoch}/{self.epochs}. Iter: {batch_idx+1}/{len(self.train_loader)}. "
                    f"LR: {lr:.6f}. Loss: {loss_value:.6f}"
                )

    def _train_step(self, batch):
        imgs, targets, _ = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(imgs)
        loss = self.criterion(logits, targets.long())
        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()
        self.seg_metrics.reset()
        self.meters.reset()

        if self.is_master:
            self.logger.info(f"Validating epoch {epoch}")
            p_bar = tqdm(self.val_loader)
        else:
            p_bar = self.val_loader

        for batch_idx, batch in enumerate(p_bar):
            val_loss = self._val_step(batch)
            try:
                val_value = val_loss.item()
            except Exception:
                val_value = float(val_loss)
            self.meters.update('val_loss', val_value)

            if self.is_master:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler else 0.0
                p_bar.set_description(
                    f"Val Epoch: {epoch}/{self.epochs}. Iter: {batch_idx+1}/{len(self.val_loader)}. "
                    f"LR: {lr:.6f}. Val Loss: {val_value:.6f}"
                )

        # Compute metrics
        avg_metrics, mc_metrics = self.seg_metrics.compute()
        if self.is_master:
            f1_avg = avg_metrics.get('f1_score', torch.tensor(0., device=self.device))
            jaccard_avg = avg_metrics.get('jaccard_index', torch.tensor(0., device=self.device))
            miou_avg = avg_metrics.get('mIOU', torch.tensor(0., device=self.device))

            self.logger.info(
                f"[Val] F1(avg): {f1_avg.item():.4f}, "
                f"Jaccard(avg): {jaccard_avg.item():.4f}, "
                f"mIoU(avg): {miou_avg.item():.4f}"
            )

            mc_f1 = mc_metrics.get('f1_score', torch.tensor([])).tolist()
            mc_jaccard = mc_metrics.get('jaccard_index', torch.tensor([])).tolist()
            mc_miou = mc_metrics.get('mIOU', torch.tensor([])).tolist()

            self.logger.info(f"[Val] F1(per-class): {', '.join(f'{v:.4f}' for v in mc_f1)}")
            self.logger.info(f"[Val] Jaccard(per-class): {', '.join(f'{v:.4f}' for v in mc_jaccard)}")
            self.logger.info(f"[Val] mIoU(per-class): {', '.join(f'{v:.4f}' for v in mc_miou)}")

        if dist.is_initialized():
            dist.barrier()

        return self.meters['val_loss'].avg

    @torch.no_grad()
    def _val_step(self, batch):
        imgs, targets, _ = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(imgs)
        val_loss = self.criterion(logits, targets.long())
        preds = torch.argmax(logits, dim=1)
        self.seg_metrics.update(preds, targets)

        return val_loss


class ObjTrainer(BaseTrainer):
    def __init__(self, *args, is_master=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_metrics = ObjectDetectionMetricLogger(device=self.device)
        self.is_master = is_master  # only rank 0 logs/saves

    def _log_metric(self, prefix: str, key: str, val):
        import torch
        if isinstance(val, dict):
            for kk, vv in val.items():
                self._log_metric(prefix, f"{key}/{kk}", vv)
            return
        if torch.is_tensor(val):
            if val.numel() == 1:
                self.logger.info(f"{prefix}{key}: {val.item():.4f}")
            else:
                self.logger.info(f"{prefix}{key}: {val.detach().cpu().tolist()}")
            return
        if isinstance(val, (float, int)):
            self.logger.info(f"{prefix}{key}: {val:.4f}")
        else:
            self.logger.info(f"{prefix}{key}: {val}")

    def _train_epoch(self, epoch):
        self.model.train()
        self.meters.reset()
        if self.is_master:
            self.logger.info(f"Training epoch {epoch}")
            p_bar = tqdm(range(len(self.train_loader)))
        else:
            p_bar = range(len(self.train_loader))

        iter_loader = iter(self.train_loader)
        for batch_idx in range(len(self.train_loader)):
            self.model.zero_grad()
            batch = next(iter_loader)
            train_loss = self._train_step(batch)
            train_loss.backward()
            self.optimizer.step()
            self.meters.update('train_loss', train_loss.item())

            if self.is_master:
                p_bar.set_description(
                    f"Train Epoch: {epoch}/{self.epochs:4}. Iter: {batch_idx + 1}/{len(self.train_loader):4}. "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}. Train Loss: {train_loss.item():.6f}"
                )
                p_bar.update()
        if self.is_master:
            p_bar.close()

    def _train_step(self, batch):
        imgs, targets, img_ids = batch
        imgs = [img.to(self.device) for img in imgs]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        loss_dict = self.model(imgs, targets)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()
        self.obj_metrics.reset()

        if 'val_loss' not in self.meters.meters:
            self.meters.meters['val_loss'] = AverageMeter()
        else:
            self.meters['val_loss'].reset()

        if self.is_master:
            self.logger.info(f"Validating epoch {epoch}")
            p_bar = tqdm(range(len(self.val_loader)))
        else:
            p_bar = range(len(self.val_loader))

        iter_loader = iter(self.val_loader)
        for batch_idx in range(len(self.val_loader)):
            batch = next(iter_loader)
            val_loss = self._val_step(batch)
            self.meters.update('val_loss', val_loss)

            if self.is_master:
                p_bar.set_description(
                    f"Val Epoch: {epoch}/{self.epochs:4}. Iter: {batch_idx + 1}/{len(self.val_loader):4}. "
                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}. Val Loss: {val_loss:.6f}"
                )
                p_bar.update()
        if self.is_master:
            p_bar.close()
            self.logger.info(f"Epoch {epoch}: Avg Validation Loss: {self.meters['val_loss'].avg:.6f}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        metrics = self.obj_metrics.compute()

        if self.is_master:
            for k, v in metrics.items():
                self._log_metric("[Val] ", k, v)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return self.meters['val_loss'].avg


    @torch.no_grad()
    def _val_step(self, batch):
        self.model.eval()
        images, targets, _ = batch
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        outputs = self.model(images)

        outputs = [{k: v.to(self.device) for k, v in out.items()} for out in outputs]

        self.obj_metrics.update(outputs, targets)

        self.model.train()
        loss_dict = self.model(images, targets)
        self.model.eval()

        if isinstance(loss_dict, dict):
            loss = sum(loss for loss in loss_dict.values())
        else:
            loss = torch.tensor(loss_dict, device=self.device)

        return loss.item()



def move_to_device(obj, device):
    """ Recursive function to move targets to device """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(o, device) for o in obj]
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    else:
        return obj
