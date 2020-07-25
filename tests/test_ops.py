import pytest
import pytorch_lightning
import torch

from src.model.ops import add, multiply, subtract


@pytest.fixture(scope="module")
def fix_seed():
    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test_multiply(fix_seed):
    result = multiply(2, 3)
    assert result == 6


def test_add(fix_seed):
    result = add(2, 3)
    assert result == 5


def test_subtract(fix_seed):
    result = subtract(3, 2)
    assert result == 1
