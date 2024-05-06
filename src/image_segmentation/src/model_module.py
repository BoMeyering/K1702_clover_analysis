from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
import numpy as np
from pytorch_lightning import LightningModule
from model import create_model

from torch.optim.lr_scheduler import ExponentialLR

class SegformerModel(LightningModule):
    def __init__(
            self,
            num_classes,
            img_size,
            model_variant,
            lr=0.001,
            gamma=0.8
    ):
        super().__init__()
        self.num_classes = num_classes
        self.model_variant = model_variant
        self.lr = lr
        self.gamma = gamma
        self.model = create_model(
            num_classes=self.num_classes,
            variant=self.model_variant
        )


    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        pixel_values = batch['pixel_values'].squeeze()
        labels = batch['labels'].squeeze()
        output = self.model(pixel_values, labels)
        loss = output['loss']
        
        logging_losses = {
            "cross_entropy_loss": loss.detach()
        }

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=True
        )

        return loss
    
    # def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     return super().validation_step(*args, **kwargs)
    
    # def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT | None:
    #     return super().test_step(*args, **kwargs)
    
    def configure_optimizers(self) -> Any:
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.lr
        )
        scheduler = ExponentialLR(
            optimizer=optimizer,
            gamma=self.gamma
        )

        return [optimizer], [scheduler]

    
