import numpy as np
import pandas as pd
from scipy.stats import chi2

from .IRR_result import IRR_result


def stuart_maxwell_mh(ratings):
    """Calculates the Stuart-Maxwell coefficient of concordance for two raters.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

    Returns
    -------
    IRR_result
        Stuart-Maxwell coefficient in an IRR_result dataclass.

    """
    ratings = pd.DataFrame(ratings)

    smx = pd.crosstab(ratings.iloc[:, 0], ratings.iloc[:, 1])
    smx = np.array(smx)

    rowsums = np.sum(smx, axis=1)
    colsums = np.sum(smx, axis=0)
    equalsums = rowsums == colsums

    if np.any(equalsums):
        smx = smx[~equalsums, ~equalsums]
        if smx.shape[0] < 2:
            raise Exception("Too many equal marginals, cannot compute")
        rowsums = np.sum(smx, axis=1)
        colsums = np.sum(smx, axis=0)

    k_minus1 = len(rowsums) - 1
    smd = (rowsums - colsums)[:k_minus1]
    smS = np.zeros((k_minus1, k_minus1))

    for i in range(k_minus1):
        for j in range(k_minus1):
            if i == j:
                smS[i, j] = rowsums[i] + colsums[j] - 2 * smx[i, j]
            else:
                smS[i, j] = -(smx[i, j] + smx[j, i])

    smstat = smd.T @ np.linalg.inv(smS) @ smd

    p = 1 - chi2.cdf(smstat, k_minus1)

    return IRR_result("Stuart-Maxwell marginal homogeneity", int(np.sum(smx)), 2, "Chisq", smstat, smstat,
                      f"Chisq({k_minus1})", p)

