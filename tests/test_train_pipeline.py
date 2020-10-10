import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger

from src.runner.runner import Runner
from src.utils import get_data_loaders
from tests.common import fix_seed, tearup_config
from train import build_model


def test_train_pipeline(fix_seed, tearup_config):
    config = OmegaConf.create(tearup_config)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    lr_logger = LearningRateLogger()
    model = build_model(model_conf=config.model)
    runner = Runner(model=model, config=config.runner)

    trainer = Trainer(
        distributed_backend=config.runner.trainer.distributed_backend,
        fast_dev_run=False,
        gpus=None,
        amp_level="O2",
        row_log_interval=10,
        callbacks=[lr_logger],
        max_epochs=1,
        weights_summary="top",
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=None,
        benchmark=False,
        deterministic=True,
        num_sanity_val_steps=5,
        overfit_batches=0.0,
        precision=32,
        profiler=True,
    )

    trainer.fit(model=runner, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
