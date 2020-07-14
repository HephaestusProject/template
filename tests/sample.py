import pytest


@pytest.fixture
def square_10():
    return 10 * 10


def test_square(square_10):
    assert square_10 == 100
