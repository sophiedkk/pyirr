from pyirr import stuart_maxwell_mh


def test_stuart_maxwell_mh(vision):
    # Example from Stuart (1955)
    mh = stuart_maxwell_mh(vision)

    assert mh.subjects == 7477
    assert mh.raters == 2
    assert round(mh.value) == 12
    assert round(mh.pvalue, 5) == 0.00753
