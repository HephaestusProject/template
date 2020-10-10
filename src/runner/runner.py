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

        result = pl.TrainResult(loss)
        result.log(name="train_loss", value=loss)
        result.log_dict(
            {"train_loss": loss, "train_acc": acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss, acc = self._comm_step(x, y)

        result = pl.EvalResult(checkpoint_on=acc)
        result.log_dict(
            {"val_loss": loss, "val_acc": acc},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return result

    def test_step(self, batch, batch_idx):
        result = self.validation_step(batch, batch_idx)
        result.rename_keys({"val_loss": "loss", "val_acc": "acc"})
        return result
