import pytest

from pytest_sample import square_10


def test_square():
    f1 = square_10()
    assert f1 == 100
