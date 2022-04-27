import pytest

from pyirr import bhapkar


def test_bhapkar(vision):
    # Original example used from Bhapkar (1966)
    bhapkar_result = bhapkar(vision)

    assert bhapkar_result.subjects == 7477
    assert bhapkar_result.raters == 2
    assert round(bhapkar_result.statistic) == 12
    assert round(bhapkar_result.pvalue, 3) == 0.007

    with pytest.raises(ValueError) as excinfo:
        bhapkar([{"Rater 1": 0, "Rater 2": 0, "Rater 3": 0}])  # too many raters

    assert "Number of raters exceeds 2" in str(excinfo.value)
