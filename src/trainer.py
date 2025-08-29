"""
src.trainer.py
Trainer Classes
BoMeyering 2025
"""
import logging
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.eval import AverageMeter, AverageMeterSet
from src.callbacks import ModelCheckpoint
from abc import ABC, abstractmethod
from src.metrics import ObjectDetectionMetricLogger
from src.metrics import SegmentationMetricLogger

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

    def train(self):
        """ Main train method """
        self.logger.info(f"Training model for {self.epochs} epochs.")
        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

            logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(self.meters['train_loss'].avg),
                "val_loss": torch.tensor(self.meters['val_loss'].avg),
                "model_state_dict": self.model.state_dict(),
            }

            self.logger.info(f"Epoch {epoch} - Train Loss: {self.meters['train_loss'].avg:.6f}, Val Loss: {self.meters['val_loss'].avg:.6f}")

            self.checkpoint(epoch=epoch, logs=logs)

            if self.scheduler:
                self.scheduler.step()

        self.logger.info(f"Training complete")

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
    def __init__(self, *args, criterion, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.seg_metrics = SegmentationMetricLogger(num_classes= 3 , device=self.device)

    def _train_epoch(self, epoch):
        """ Train one epoch """
        # Put model in training mode
        self.model.train()
        self.meters.reset()

        self.logger.info(f"Training epoch {epoch}")

        p_bar = tqdm(range(len(self.train_loader)))

        iter_loader = iter(self.train_loader)

        for batch_idx in range(len(self.train_loader)):
            # Zero the gradient
            self.model.zero_grad()
                
            # Grab the next batch and run through train_step
            batch = next(iter_loader)
            train_loss = self._train_step(batch)

            # Backpropagate the errors
            train_loss.backward()

            # Update the train loss meter
            self.meters.update('train_loss', train_loss.item())

            # Step the optimizer
            self.optimizer.step()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Train Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=train_loss.item()
                )
            )
            p_bar.update()
        p_bar.close()

    def _train_step(self, batch):
        """ Train one batch """
        
        # Unpack the batch
        imgs, targets, img_ids = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass through the model and compute loss
        logits = self.model(imgs)

        # upsampled_logits = F.interpolate(output['logits'], size=(512, 512), mode="bilinear", align_corners=False)

        train_loss = self.criterion(logits, targets.long())

        return train_loss
    
    @torch.no_grad()
    def _val_epoch(self, epoch):
        """ Validate one epoch """

        self.seg_metrics.reset()

        # Put model in evaluation mode
        self.model.eval()

        self.logger.info(f"Validating epoch {epoch}")

        p_bar = tqdm(range(len(self.val_loader)))

        iter_loader = iter(self.val_loader)

        for batch_idx in range(len(self.val_loader)):
            batch = next(iter_loader)
            val_loss = self._val_step(batch)
            
            # Update the val loss meter
            self.meters.update('val_loss', val_loss.item())

            # Update progress bar
            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Val Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.val_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=val_loss.item()
                )
            )
            p_bar.update()
        p_bar.close()
        
        #logging avg_metrics
        avg_metrics, mc_metrics = self.seg_metrics.compute()
        self.logger.info(
            f"[Val] F1 Score (avg): {avg_metrics.get('f1_score', torch.tensor(0.)).item():.4f}, "
            f"Jaccard Index (avg): {avg_metrics.get('jaccard_index', torch.tensor(0.)).item():.4f}, "
            f"Mean IoU (avg): {avg_metrics.get('mIOU', torch.tensor(0.)).item():.4f}"
        )

        # Now log multiclass metrics - converting tensors to list because we cannot log tensors we need it to look reasonable
        mc_f1 = mc_metrics.get('f1_score', torch.tensor([])).tolist()
        mc_jaccard = mc_metrics.get('jaccard_index', torch.tensor([])).tolist()
        mc_miou = mc_metrics.get('mIOU', torch.tensor([])).tolist()

        self.logger.info(
            f"[Val] F1 Score (per-class): {', '.join(f'{v:.4f}' for v in mc_f1)}"
        )
        self.logger.info(
            f"[Val] Jaccard Index (per-class): {', '.join(f'{v:.4f}' for v in mc_jaccard)}"
        )
        self.logger.info(
            f"[Val] Mean IoU (per-class): {', '.join(f'{v:.4f}' for v in mc_miou)}"
        )

    @torch.no_grad()
    def _val_step(self, batch):
        """ Validate one batch """

        # Unpack the batch
        imgs, targets, img_ids = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass through model and compute loss
        logits = self.model(imgs)

        val_loss = self.criterion(logits, targets.long())

        # Get predicted class IDs to give to the metrics function
        preds = torch.argmax(logits, dim=1)  

        # Update metrics (pls work)
        self.seg_metrics.update(preds, targets)

        return val_loss
    

class ObjTrainer(BaseTrainer):
    def __init__(self, *args, is_master=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.obj_metrics = ObjectDetectionMetricLogger(device=self.device)
        self.is_master = is_master  # only rank 0 logs/saves

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
                if torch.is_tensor(v):
                    v = v.item()
                self.logger.info(f"[Val] {k}: {v:.4f}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return self.meters['val_loss'].avg


    @torch.no_grad()
    def _val_step(self, batch):
        self.model.eval()
        images, targets, _ = batch
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        # Forward pass for predictions
        outputs = self.model(images)

        # Move outputs to device if needed (some models may produce CPU tensors)
        outputs = [{k: v.to(self.device) for k, v in out.items()} for out in outputs]

        # Update detection metrics
        self.obj_metrics.update(outputs, targets)

        # Compute loss (model needs train mode)
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