from pyirr import robinson


def test_robinson(anxiety):
    rob = robinson(anxiety)

    assert rob.subjects == 20
    assert rob.raters == 3
    assert round(rob.value, 3) == 0.477
