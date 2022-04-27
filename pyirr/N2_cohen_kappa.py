import operator

import numpy as np
from scipy.stats import norm


def max_tau(Marg, Kp):
    Kappa = Kp
    Mrg = Marg
    Dim = len(Mrg)
    PIe = np.sum(Mrg * Mrg)
    PI0 = Kappa + PIe * (1 - Kappa)

    if PI0 <= 0:
        raise Exception("Invalid set of inputs")

    Part1 = PI0 * (1 - PIe) ** 2
    Part3 = (PI0 * PIe - 2 * PIe + PI0) ** 2
    m = np.zeros((Dim, Dim))

    for ii in range(Dim):
        for jj in range(Dim):
            if ii == jj:
                m[ii, jj] = -1 * (1 - PI0) * 2 * Mrg[ii] * (2 * (1 - PIe) - (1 - PI0) * 2 * Mrg[ii])
            else:
                m[ii, jj] = (1 - PI0) ** 2 * (Mrg[ii] + Mrg[jj]) ** 2

    AA = np.diag(np.diag(np.ones((Dim ** 2, Dim ** 2))))
    BB = np.zeros((Dim, Dim ** 2))
    CC = np.diag(np.diag(np.ones((Dim, Dim))))

    for i in range(Dim):
        BB[i, (i * Dim):((i + 1) * Dim)] = 1.
        if i < Dim - 1:
            CC = np.hstack([CC, np.diag(np.diag(np.ones((Dim, Dim))))])

    DD = np.ones(Dim ** 2)
    EE = np.diag(np.diag(np.ones((Dim, Dim)))).ravel()
    A = np.vstack([AA, BB, CC, DD, EE])

    f_con = A
    f_obj = m.ravel()
    f_dir = [operator.ge] * Dim**2 + [operator.eq] * (2 * (Dim + 1))
    f_rhs = np.hstack([np.zeros(Dim ** 2), np.tile(Mrg, 2), 1., PI0])

    Part2, _ = maximize_from_iterables(f_obj, f_con, f_dir, f_rhs)

    TauSq = Part1 + Part2 - Part3
    if TauSq <= 0:
        raise Exception("Invalid set of inputs")
    Tau = np.sqrt(TauSq) / (1 - PIe) ** 2
    return Tau


def maximize_from_iterables(objectives, constraints, directions, rh_constraints):
    try:
        from pulp import lpDot, lpSum, LpProblem, LpMaximize, LpVariable, value
    except ModuleNotFoundError:
        raise ImportError("This function requires the pulp library, please install that library first.")

    prob = LpProblem("myProblem", LpMaximize)

    variable_list = []
    for i, objective in enumerate(objectives):
        variable_list.append(LpVariable("x" + str(i).zfill(3)))

    for constraint, direction, rh_constraint in zip(constraints, directions, rh_constraints):
        prob += direction(lpDot(constraint, variable_list), rh_constraint)

    prob += lpSum(lpDot(objectives, variable_list))
    prob.solve()

    result = 0
    values = []
    for v, m in zip(variable_list, objectives):
        result += value(v) * m
        values.append(v.varValue)
    return result, values


def N2_cohen_kappa(mrg, k1, k0, alpha=0.05, power=0.8, twosided=False):
    """Sample Size Calculation for Cohen's Kappa Statistic with more than one category

    Parameters
    ----------
    mrg: array-like
        a vector of marginal probabilities given by raters
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

    mrg = np.array(mrg)

    if np.any(mrg < 0):
        raise Exception("At least one marginal probability is negative")
    if np.abs(k0) > 1:
        raise Exception("Invalid Value for Kappa under H0")
    if np.abs(k1) > 1:
        raise Exception("Invalid Value for Kappa under H1")
    if not np.abs(k1 - k0) > 0:
        raise Exception("Kappa under H0 must be different from H1")
    twosided = 2 if twosided else 1
    if len(mrg) >= 10:
        raise Exception("Valid only up to 10 categories")
    if len(mrg) <= 1:
        raise Exception("Valid only for greater than equal to 2 categories")
    if not np.sum(mrg) == 1:
        raise Exception("Sum of marginal probabilities is not 1")
    if alpha <= 0 or alpha >= 1:
        raise Exception("Alpha must be between 0 and 1")
    if power <= 0 or power >= 1:
        raise Exception("Power must be between 0 and 1")
    if not power-alpha > 0:
        raise Exception("Power is less than Alpha")

    tau_null = max_tau(mrg, k0)
    tau_alt = max_tau(mrg, k1)
    z_alpha = norm.ppf(1 - alpha / twosided)
    z_beta = norm.ppf(power)
    raw_n = (((z_alpha * tau_null) + (z_beta * tau_alt)) / (k0 - k1))**2
    n = np.ceil(raw_n)

    return int(n)
