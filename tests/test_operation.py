import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lib.operation import multiply, add


def test_multiply():
    result = multiply(2, 3)
    assert result == 6

def test_add():
    result = add(2, 3)
    assert result == 5
