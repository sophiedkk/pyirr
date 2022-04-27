from pyirr import rater_bias


def test_rater_bias(vision):
    # Example from Bishop, Fienberg & Holland (1978), Table 8.2-1
    bias = rater_bias(vision)

    assert bias.subjects == 7477
    assert bias.raters == 2
    assert round(bias.value, 3) == 0.537
    assert round(bias.statistic, 1) == 11.9
    assert round(bias.pvalue, 6) == 0.000566
