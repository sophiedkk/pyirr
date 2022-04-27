from math import ceil

from scipy.stats import norm


def N_cohen_kappa(rate1, rate2, k1, k0, alpha=0.05, power=0.8, twosided=False):
    """Sample Size Calculation for Cohen's Kappa Statistic.

    Parameters
    ----------
    rate1: float
        the probability that the first rater will record a positive diagnosis
    rate2: float
        the probability that the second rater will record a positive diagnosis
    k1: float
        the true Cohen's Kappa statistic
    k0: float
        the value of Kappa under the null hypothesis
    alpha: float
        type I error of data
    power: float
        the desired power to detect the difference between true Kappa and hypothetical Kappa
    twosided: bool
        True if test is two-sided

    Returns
    -------
    int
        Sample size
    """

    d = 2 if twosided else 1

    pi2 = 1 - rate1
    pi_2 = 1 - rate2

    pie = rate1 * rate2 + pi2 * pi_2
    pi0 = k1 * (1 - pie) + pie

    pi22 = (pi0 - rate1 + pi_2) / 2
    pi11 = pi0 - pi22
    pi12 = rate1 - pi11
    pi21 = rate2 - pi11

    pi0_h = k0 * (1 - pie) + pie

    pi22_h = (pi0_h - rate1 + pi_2) / 2
    pi11_h = pi0_h - pi22_h
    pi12_h = rate1 - pi11_h
    pi21_h = rate2 - pi11_h

    Q = (1 - pie)**-4 * (pi11 * (1 - pie - (rate2 + rate1) * ( 1 - pi0))**2 + pi22 * (1 - pie - (pi_2 + pi2) *
        (1 - pi0))**2 + (1 - pi0)**2 * (pi12 * (rate2 + pi2)**2 + pi21 * (pi_2 + rate1)**2) - (pi0 * pie - 2 *
        pie + pi0)**2)

    Q_h = (1 - pie)**-4 * (pi11_h * (1 - pie - (rate2 + rate1) * (1 - pi0_h))**2 + pi22_h * (1 - pie - (pi_2 + pi2) *
          (1 - pi0_h))**2 + (1 - pi0_h)**2 * (pi12_h * (rate2 + pi2)**2 + pi21_h * (pi2 + rate1)**2) - (pi0_h * pie -
          2 * pie + pi0_h)**2)

    N = ((norm.ppf(1 - alpha / d) * Q_h**0.5 + norm.ppf(power) * Q**0.5) / (k1 - k0))**2

    return ceil(N)
