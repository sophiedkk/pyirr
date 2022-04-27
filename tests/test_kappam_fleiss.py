from pyirr import kappam_fleiss


def test_kappam_fleiss(diagnoses):
    kappam = kappam_fleiss(diagnoses)

    assert kappam.subjects == 30
    assert kappam.raters == 6
    assert round(kappam.value, 3) == 0.430
    assert round(kappam.statistic, 1) == 17.7
    assert round(kappam.pvalue, 3) == 0.000


def test_kappam_fleiss_exact(diagnoses):
    kappam = kappam_fleiss(diagnoses, exact=True)
    assert round(kappam.value, 3) == 0.442


def test_kappam_fleiss_detail(diagnoses):
    kappam = kappam_fleiss(diagnoses, detail=True)
    expected = [0.245, 0.245, 0.520, 0.471, 0.566]

    assert list(kappam.detail.Kappa) == expected
