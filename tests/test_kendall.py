from pyirr import kendall


def test_kendall(anxiety):
    kendall_result = kendall(anxiety, True)

    assert kendall_result.subjects == 20
    assert kendall_result.raters == 3
    assert round(kendall_result.value, 2) == 0.54
    assert round(kendall_result.statistic, 1) == 30.8
    assert round(kendall_result.pvalue, 4) == 0.0429
