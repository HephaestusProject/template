import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf

from src.model.net import LeNet
from tests.common import fix_seed, tearup_config

forward_test_case = [
    # (device, test_input)
    ("cpu", torch.randn(((2, 1, 32, 32)))),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.randn(((2, 1, 32, 32))),
    ),
]


@pytest.mark.parametrize(
    "device, test_input",
    forward_test_case,
)
def test_binarylinear_forward(
    fix_seed,
    tearup_config,
    device,
    test_input,
):
    config = tearup_config
    model = LeNet(config.model).to(device)

    test_input = test_input.to(device)
    model(test_input)


summary_test_case = [
    # (device, test_input)
    ("cpu"),
    (torch.device("cuda" if torch.cuda.is_available() else "cpu")),
]


@pytest.mark.parametrize("device", summary_test_case)
def test_binarylinear_summary(fix_seed, tearup_config, device):
    config = tearup_config
    model = LeNet(config.model).to(device=device)
    model.summary()
