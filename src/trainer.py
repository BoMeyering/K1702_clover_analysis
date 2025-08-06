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

    def _train_epoch(self, epoch):
        self.model.train()
        self.meters.reset()

        self.logger.info(f"Training epoch {epoch}")
        p_bar = tqdm(range(len(self.train_loader)))
        iter_loader = iter(self.train_loader)

        for batch_idx in range(len(self.train_loader)):
            self.model.zero_grad()
            batch = next(iter_loader)
            train_loss = self._train_step(batch)
            train_loss.backward()
            self.meters.update('train_loss', train_loss.item())
            self.optimizer.step()

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
        imgs, targets, img_ids = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(imgs)
        train_loss = self.criterion(logits, targets.long())
        return train_loss

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()
        self.logger.info(f"Validating epoch {epoch}")
        p_bar = tqdm(range(len(self.val_loader)))
        iter_loader = iter(self.val_loader)

        for batch_idx in range(len(self.val_loader)):
            batch = next(iter_loader)
            val_loss = self._val_step(batch)
            self.meters.update('val_loss', val_loss.item())

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

    @torch.no_grad()
    def _val_step(self, batch):
        imgs, targets, img_ids = batch
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(imgs)
        val_loss = self.criterion(logits, targets.long())
        return val_loss


class ObjTrainer(BaseTrainer):
    def _train_epoch(self, epoch):
        self.model.train()
        self.meters.reset()

        self.logger.info(f"Training epoch {epoch}")
        p_bar = tqdm(range(len(self.train_loader)))
        iter_loader = iter(self.train_loader)

        for batch_idx in range(len(self.train_loader)):
            self.model.zero_grad()
            batch = next(iter_loader)
            train_loss = self._train_step(batch)
            train_loss.backward()
            self.meters.update('train_loss', train_loss.item())
            self.optimizer.step()

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
        self.model.train()
        imgs, targets, img_id = batch
        imgs = imgs.to(self.device)
        loss_dict = self.model(imgs, targets)
        loss = loss_dict["loss"]
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch):
        self.model.eval()
        self.logger.info(f"Validating epoch {epoch}")
        p_bar = tqdm(range(len(self.val_loader)))
        iter_loader = iter(self.val_loader)

        for batch_idx in range(len(self.val_loader)):
            batch = next(iter_loader)
            val_loss = self._val_step(batch)
            self.meters.update('val_loss', val_loss.item())

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

    @torch.no_grad()
    def _val_step(self, batch):
        imgs, targets, img_ids = batch
        imgs = imgs.to(self.device)
        targets = {k: [j.to(self.device) for j in v] for k, v in targets.items()}
        loss_dict = self.model(imgs, targets)
        loss = loss_dict["loss"]
        return loss
