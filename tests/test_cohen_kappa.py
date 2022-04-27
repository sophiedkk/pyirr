from pyirr import N_cohen_kappa, N2_cohen_kappa


def test_n_cohen_kappa():
    n = N_cohen_kappa(0.5, 0.5, 0.7, 0.85)
    assert n == 96

    n = N2_cohen_kappa([0.2, 0.25, 0.55], k1=0.6, k0=0.4)
    assert n == 101

    n = N2_cohen_kappa([0.2, 0.25, 0.55], k1=0.6, k0=0.4, power=0.9)
    assert n == 136

    n = N2_cohen_kappa([0.2, 0.05, 0.2, 0.05, 0.2, 0.3], k1=0.5, k0=0.1)
    assert n == 18
