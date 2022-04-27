import pytest

from pyirr import read_data


@pytest.fixture
def video():
    return read_data("video")


@pytest.fixture
def vision():
    return read_data("vision")


@pytest.fixture
def anxiety():
    return read_data("anxiety")
