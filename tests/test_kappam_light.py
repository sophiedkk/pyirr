from pyirr import kappam_light
from pyirr.IRR_result import IRR_result


def test_kappam_light(diagnoses):
    kappam = kappam_light(diagnoses)

    assert kappam.subjects == 30
    assert kappam.raters == 6
    assert round(kappam.value, 3) == 0.459
    assert round(kappam.statistic, 2) == 2.31
    assert round(kappam.pvalue, 4) == 0.0211
