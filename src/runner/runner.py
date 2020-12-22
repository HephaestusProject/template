from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning.core import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import optim
from torch.optim.lr_scheduler import StepLR

from src.utils import load_class


class Runner(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        super().__init__()
        self.model = model
        self.hparams.update(config)
        self.config = config
        print(self.hparams)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        args = dict(self.config.optimizer.params)
        args.update({"params": self.model.parameters()})

        opt = load_class(module=optim, name=self.config.optimizer.type, args=args)
        scheduler = StepLR(optimizer=opt, step_size=10)

        return [opt], [scheduler]

    def _comm_step(self, x, y):
        y_hat = self(x)
        loss = self.model.loss(y_hat, y)

        pred = torch.argmax(y_hat, dim=1)
        acc = accuracy(pred, y)

        return y_hat, loss, acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self._comm_step(x, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self._comm_step(x, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {"val_loss": loss, "val_acc": acc}
