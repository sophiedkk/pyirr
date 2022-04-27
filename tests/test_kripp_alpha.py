import numpy as np

from pyirr import kripp_alpha


def test_kripp_alpha():
    nmm = np.array([1, 1, np.NaN, 1, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 1, 2, 3, 4, 4, 4, 4, 4,
                    1, 1, 2, 1, 2, 2, 2, 2, np.NaN, 5, 5, 5, np.NaN, np.NaN, 1, 1, np.NaN, np.NaN, 3, np.NaN])
    nmm = nmm.reshape((-1, 4))

    kripp_nominal = kripp_alpha(nmm, method="nominal")
    assert kripp_nominal.subjects == 12
    assert kripp_nominal.raters == 4
    assert round(kripp_nominal.value, 3) == 0.743

    kripp_ordinal = kripp_alpha(nmm, method="ordinal")
    assert round(kripp_ordinal.value, 3) == 0.815

    kripp_interval = kripp_alpha(nmm, method="interval")
    assert round(kripp_interval.value, 3) == 0.849

    kripp_ratio = kripp_alpha(nmm, method="ratio")
    assert round(kripp_ratio.value, 3) == 0.797
