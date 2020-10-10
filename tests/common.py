import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="module")
def tearup_config():
    return OmegaConf.create(
        {
            "dataset": {
                "type": "MNIST",
                "params": {
                    "root": "data/",
                    "train": None,
                    "transform": None,
                    "target_transform": None,
                    "download": True,
                },
            },
            "model": {
                "type": "LeNet",
                "params": {
                    "width": 32,
                    "height": 32,
                    "channels": 1,
                    "classes": 10,
                    "feature_layers": {
                        "conv": [
                            {
                                "in_channels": 1,
                                "out_channels": 6,
                                "kernel_size": 5,
                                "stride": 1,
                                "padding": 0,
                                "bias": True,
                                "padding_mode": "zeros",
                                "activation": {"type": "Tanh", "args": {}},
                                "pool": {
                                    "type": "AvgPool2d",
                                    "args": {"kernel_size": [2, 2], "padding": 0},
                                },
                            },
                            {
                                "in_channels": 6,
                                "out_channels": 16,
                                "kernel_size": 5,
                                "stride": 1,
                                "padding": 0,
                                "bias": True,
                                "padding_mode": "zeros",
                                "activation": {"type": "Tanh", "args": {}},
                                "pool": {
                                    "type": "AvgPool2d",
                                    "args": {"kernel_size": [2, 2], "padding": 0},
                                },
                            },
                            {
                                "in_channels": 16,
                                "out_channels": 120,
                                "kernel_size": 5,
                                "stride": 1,
                                "padding": 0,
                                "bias": True,
                                "padding_mode": "zeros",
                                "activation": {"type": "Tanh", "args": {}},
                                "pool": None,
                            },
                        ],
                        "linear": [
                            {
                                "in_features": 120,
                                "out_features": 84,
                                "bias": True,
                                "activation": {"type": "Tanh", "args": {}},
                            },
                            {
                                "in_features": 84,
                                "out_features": 10,
                                "bias": True,
                                "activation": None,
                            },
                        ],
                    },
                    "output_layer": {"type": "Softmax", "args": {"dim": 1}},
                },
            },
            "runner": {
                "type": "Runner",
                "dataloader": {
                    "type": "DataLoader",
                    "params": {"num_workers": 48, "batch_size": 256},
                },
                "optimizer": {"type": "SGD", "params": {"lr": 0.01, "momentum": 0}},
                "trainer": {
                    "type": "Trainer",
                    "params": {
                        "max_epochs": 15,
                        "gpus": -1,
                        "distributed_backend": "ddp",
                        "fast_dev_run": False,
                        "amp_level": "02",
                        "row_log_interval": 10,
                        "weights_summary": "top",
                        "reload_dataloaders_every_epoch": False,
                        "resume_from_checkpoint": None,
                        "benchmark": False,
                        "deterministic": True,
                        "num_sanity_val_steps": 5,
                        "overfit_batches": 0.0,
                        "precision": 32,
                        "profiler": True,
                    },
                },
                "experiments": {"name": "renet", "output_dir": "output/"},
            },
        }
    )
