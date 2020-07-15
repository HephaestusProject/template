import sys
import os
import pytest
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from lib.operation import multiply


def test_multiply():
    result = multiply(2, 3)
    assert result == 6
