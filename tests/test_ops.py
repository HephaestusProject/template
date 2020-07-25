import pytest
import pytorch_lightning
import torch

from src.model.ops import add, multiply, subtract


def test_multiply():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    result = multiply(2, 3)
    assert result == 6


def test_add():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    result = add(2, 3)
    assert result == 5


def test_subtract():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    result = subtract(3, 2)
    assert result == 1
