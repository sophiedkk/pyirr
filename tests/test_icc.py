import numpy as np

from pyirr import intraclass_correlation


def test_intraclass_correlation(anxiety):
    icc = intraclass_correlation(anxiety, model="twoway", mtype="agreement")

    assert icc.model == "twoway"
    assert icc.mtype == "agreement"
    assert icc.subjects == 20
    assert icc.raters == 3
    assert round(icc.value, 3) == 0.198
    assert round(icc.Fvalue, 2) == 1.83
    assert round(icc.pvalue, 4) == 0.0543
    assert icc.df1 == 19
    assert round(icc.df2, 1) == 39.7
    assert round(icc.lower_bound, 3) == -0.039
    assert round(icc.upper_bound, 3) == 0.494


def test_intraclass_correlation_mtype():
    row_1 = [5, 12, 11, 10, 9, 7, 5, 12, 12, 7, 12, 17, 15, 13, 6, 13, 20, 14, 13, 11]
    row_2 = [14, 22, 21, 19, 22, 16, 13, 24, 20, 19, 26, 22, 26, 18, 10, 26, 30, 22, 23, 23]
    row_3 = [27, 31, 29, 29, 32, 23, 29, 28, 30, 28, 31, 35, 34, 32, 30, 29, 39, 37, 33, 29]
    data = np.array([row_1, row_2, row_3]).T

    consistency = intraclass_correlation(data, "twoway")  # High consistency
    assert round(consistency.value, 3) == 0.695

    agreement = intraclass_correlation(data, "twoway", "agreement")  # Low agreement
    assert round(agreement.value, 3) == 0.108
