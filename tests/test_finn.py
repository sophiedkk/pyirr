import pytest

from pyirr import finn


def test_finn(video):
    finn_result = finn(video, 6, model="twoway")

    assert finn_result.subjects == 20
    assert finn_result.raters == 4
    assert round(finn_result.value, 3) == 0.925
    assert round(finn_result.statistic, 1) == 13.3
    assert round(finn_result.pvalue) == 0

    with pytest.raises(ValueError) as excinfo:
        finn(video, 6, model="threeway")  # should be oneway or twoway

    assert "Model should be either 'oneway' or 'twoway'." in str(excinfo.value)
