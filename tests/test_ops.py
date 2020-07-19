from src.model.ops import multiply, add, subtract


def test_multiply():
    result = multiply(2, 3)
    assert result == 6


def test_add():
    result = add(2, 3)
    assert result == 5


def test_subtract():
    result = subtract(3, 2)
    assert result == 1
