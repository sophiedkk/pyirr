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


@pytest.fixture
def diagnoses():
    return read_data("diagnoses")


@pytest.fixture
def photo():
    return read_data("photo")


@pytest.fixture
def gonio():
    return read_data("gonio")
