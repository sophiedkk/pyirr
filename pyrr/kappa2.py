from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class kappa2_result:
    subjects: int
    raters: int
    value: float
    weight: str

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + f"Cohen's Kappa for 2 Raters (Weights: {self.weight})".center(50, " ")
        model_string += "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\nA = {self.value:.3f}\n" + "=" * 50
        return model_string


def kappa2(ratings, weight, sort_levels=False):
    """Cohen’s Kappa and weighted Kappa for two raters

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    weight: {"unweighted", "equal", "squared"}
        either a character string specifying one predefined set of weights or a numeric vector with own weights.
    sort_levels: bool

    """
    ratings = np.asarray(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("Number of raters exceeds 2. Try kappam_fleiss or kappam_light")

    r1, r2 = ratings[:, 0], ratings[:, 1]

    levels = np.unique(ratings)

    if sort_levels:
        levels = np.sort(levels)

    r1 = pd.Categorical(r1, categories=levels)
    r2 = pd.Categorical(r2, categories=levels)

    ttab = pd.crosstab(r1, r2, dropna=False)
    nc = ttab.shape[1]


    if not isinstance(weight, str):
        w = 1 - (np.asarray(weight) - min(weight)) / (max(weight) - min(weight))
    elif weight == "equal":
        w = np.arange((nc-1), -1, -1) / (nc-1)
    elif weight == "squared":
        w = 1 - np.arange(0, nc, 1)**2 / (nc-1)**2
    elif weight == "unweighted":
        w = np.zeros(nc)
        w[0] = 1

    nw = len(w)
    wvec = np.append(np.sort(w), w[1:])
    weight_tab = np.zeros((nw, nw))

    for i in range(nw):
        weight_tab[i, :] = wvec[(nw - i - 1):(2 * nw - i - 1)]

    agreeP = np.sum(ttab * weight_tab) / ns

    tm1 = np.sum(ttab, 1)
    tm2 = np.sum(ttab, 0)

    eij = np.outer(tm1, tm2) / ns
    chance_P = np.sum(eij * weight_tab) / ns

    # Kappa for 2 raters
    value = (agreeP - chance_P) / (1 - chance_P)

    # Compute statistics
    wi = np.sum(np.tile(tm2 / ns, nc) * weight_tab, 0)  # TODO: no idea what is going on here
    wj = np.sum(np.repeat(tm1 / ns, nc) * weight_tab, 1)

    var_matrix = (eij / ns) * (weight_tab - np.outer(wi, wj)) ** 2

    var_kappa = (np.sum(var_matrix) - chance_P ** 2) / (ns * (1 - chance_P) ** 2)

    SE_kappa = np.sqrt(var_kappa)
    u = value / SE_kappa

    p_value = 2 * (1 - norm.cdf(abs(u)))

    return kappa2_result(ns, nr, p_value, weight)


kappa2(pd.read_csv("../tests/anxiety.csv").iloc[:, :2], weight="equal")

# kappa2(anxiety[, 1:2], "squared")  # TODO: test cases
# Cohen
# 's Kappa for 2 Raters (Weights: squared)
#
# Subjects = 20
# Raters = 2
# Kappa = 0.297
#
# z = 1.34
# p - value = 0.18

# kappa2(diagnoses[,2:3])
#  Cohen's Kappa for 2 Raters (Weights: unweighted)
#
#  Subjects = 30
#    Raters = 2
#     Kappa = 0.631
#
#         z = 7.56
#   p-value = 4.04e-14
