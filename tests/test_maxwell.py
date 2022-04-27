from pyirr import maxwell

import pandas as pd


def test_maxwell(anxiety: pd.DataFrame):
    median_split = anxiety >= anxiety.median()
    median_split = median_split.apply(lambda col: col.astype(int))
    maxwell_result = maxwell(median_split.iloc[:, :2])

    assert maxwell_result.subjects == 20
    assert maxwell_result.raters == 2
    assert round(maxwell_result.value, 3) == 0.600
