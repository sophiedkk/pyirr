import numpy as np

from pyirr import kappa2


def test_kappa2(anxiety):
    kappa_result = kappa2(anxiety.iloc[:, :2], "squared")

    assert kappa_result.subjects == 20
    assert kappa_result.raters == 2
    assert round(kappa_result.value, 3) == 0.297
    assert round(kappa_result.statistic, 3) == 1.34
    assert round(kappa_result.pvalue, 3) == 0.180

    kappa_weights = kappa2(anxiety.iloc[:, :2], np.arange(6)**2)
    assert round(kappa_weights.value, 3) == 0.297


def test_kappa2_categorical(diagnoses):
    kappa_result = kappa2(diagnoses.iloc[:, 1:3], "unweighted")

    assert kappa_result.subjects == 30
    assert kappa_result.raters == 2
    assert round(kappa_result.value, 3) == 0.631
    assert round(kappa_result.statistic, 3) == 7.560
    assert round(kappa_result.pvalue, 3) == 0.000
