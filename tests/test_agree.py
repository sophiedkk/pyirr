from pyirr import agree


def test_agree(video):
    agreement = agree(video)

    assert agreement.raters == 4
    assert agreement.subjects == 20
    assert agreement.value == 35

    agree_with_tol = agree(video, tolerance=1)

    assert agree_with_tol.raters == 4
    assert agree_with_tol.subjects == 20
    assert agree_with_tol.value == 90
