"""
Usage:
    main.py train [options] [--dataset-config=<dataset config path>] [--model-config=<model config path>] [--runner-config=<runner config path>]
    main.py train (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/dataset/dataset.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/model/model.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/runner/runner.yml] [type: path]
    -h --help  Show this.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader

from src.model import net as Net
from src.runner.runner import Runner
from src.utils import (
    build_model,
    get_checkpoint_callback,
    get_config,
    get_data_loaders,
    get_early_stopper,
    get_log_dir,
    get_next_version,
    load_class,
)


def train(hparams: Dict):
    config_list: List = ["--dataset-config", "--model-config", "--runner-config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)

    log_dir: Path = get_log_dir(config=config)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = get_checkpoint_callback(log_dir=log_dir, config=config)

    early_stop_callback = get_early_stopper(
        early_stopping_config=config.runner.earlystopping.params
    )

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    model = build_model(model_conf=config.model)

    runner = Runner(model=model, config=config.runner)
    trainer = Trainer(
        accelerator=config.runner.trainer.distributed_backend,
        fast_dev_run=False,
        gpus=config.runner.trainer.params.gpus,
        amp_level="O2",
        callbacks=[early_stop_callback],
        checkpoint_callback=checkpoint_callback,
        max_epochs=config.runner.trainer.params.max_epochs,
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
    trainer.fit(
        model=runner,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
