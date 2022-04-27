import numpy as np

from pyirr import iota


def test_iota(diagnoses):
    iota_result = iota([diagnoses], "nominal")

    assert iota_result.subjects == 30
    assert iota_result.raters == 6
    assert round(iota_result.value, 3) == 0.442


def test_iota_standardized():
    photo = [np.array([[71, 74, 76],
                       [73, 80, 80],
                       [86, 101, 93],
                       [59, 62, 66],
                       [71, 83, 77]]),
             np.array([[166, 171, 171],
                       [160, 170, 165],
                       [187, 174, 185],
                       [161, 163, 162],
                       [172, 182, 181]])
             ]

    iota_result = iota(photo, "quantitative")
    assert round(iota_result.value, 3) == 0.755

    iota_standardized = iota(photo, "quantitative", standardize=True)
    assert round(iota_standardized.value, 3) == 0.745
