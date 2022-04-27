import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm

from .IRR_result import IRR_result


def kappa2(ratings, weight, sort_levels=False):
    """Cohen’s Kappa and weighted Kappa for two raters

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    weight: {"unweighted", "equal", "squared"} or array_like
        either a character string specifying one predefined set of weights or a numeric vector with own weights.
    sort_levels: bool

    Returns
    -------
    IRR_result
        Returns Cohen's Kappa as an IRR_result dataclass.
    """
    ratings = pd.DataFrame(ratings)  # make sure ratings is a DataFrame

    ratings.dropna(inplace=True)

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("Number of raters exceeds 2. Try kappam_fleiss or kappam_light")

    r1, r2 = ratings.iloc[:, 0], ratings.iloc[:, 1]

    if is_numeric_dtype(r1) or is_numeric_dtype(r2):
        sort_levels = True

    levels = np.unique(ratings)

    if sort_levels:
        levels = np.sort(levels)

    r1 = pd.Categorical(r1, categories=levels)
    r2 = pd.Categorical(r2, categories=levels)

    ttab = pd.crosstab(r1, r2, dropna=False).values
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
    chanceP = np.sum(eij * weight_tab) / ns

    # Kappa for 2 raters
    value = (agreeP - chanceP) / (1 - chanceP)

    # Compute statistics
    wi = np.sum(np.broadcast_to(tm2/ns, (nc, nc)).T * weight_tab, 0)
    wj = np.sum(np.repeat(tm1/ns, nc).reshape((nc, nc)).T * weight_tab, 1)

    var_matrix = (eij / ns) * (weight_tab - np.sum(np.meshgrid(wi, wj), axis=0).T) ** 2

    var_kappa = (np.sum(var_matrix) - chanceP ** 2) / (ns * (1 - chanceP) ** 2)

    SE_kappa = np.sqrt(var_kappa)
    u = value / SE_kappa

    pvalue = 2 * (1 - norm.cdf(abs(u)))

    # return kappa2_result(ns, nr, value, u, p_value, weight)
    method = f"Cohen's Kappa for 2 Raters (Weights: {weight})"
    return IRR_result(method, ns, nr, "Kappa", value, u, "z", pvalue)
