from pyirr import meanrho


def test_meanrho(anxiety):
    rho = meanrho(anxiety)

    assert rho.subjects == 20
    assert rho.raters == 3
    assert round(rho.value, 3) == 0.314
    assert round(rho.statistic, 2) == 1.29
    assert round(rho.pvalue, 3) == 0.196
    assert rho.error is not None
