from pyirr import meancor


def test_meancor(anxiety):
    cor = meancor(anxiety)

    assert cor.subjects == 20
    assert cor.raters == 3
    assert round(cor.value, 3) == 0.224
    assert round(cor.statistic, 3) == 0.922
    assert round(cor.pvalue, 3) == 0.357
