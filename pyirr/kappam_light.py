import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import norm

from .kappa2 import kappa2
from .IRR_result import IRR_result


def kappam_light(ratings):
    """Computes Light's Kappa as an index of interrater agreement between m raters on categorical data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

    Returns
    -------
    IRR_result
        Returns Light's Kappa as an IRR_result dataclass.

   """
    ratings = pd.DataFrame(ratings)
    ns = ratings.shape[0]
    nr = ratings.shape[1]

    ratings.dropna(inplace=True)  # drop nans

    kappas = []
    for i in range(nr - 1):
        for j in range(i + 1, nr):
            kappas.append(kappa2(ratings.iloc[:, [i, j]], weight="unweighted").value)

    value = np.mean(kappas)

    # Variance & Computation of p-value
    lev = np.unique(ratings)
    levlen = len(lev)

    dis = []
    disrater = []
    for nri in range(nr - 1):
        for nrj in range(nri + 1, nr):
            for i in range(levlen):
                for j in range(levlen):
                    if i != j:
                        r1i = np.sum(ratings.iloc[:, nri] == lev[i])
                        r2j = np.sum(ratings.iloc[:, nrj] == lev[j])
                        dis.append(r1i * r2j)
            disrater.append(np.sum(dis))
            dis = []  # reset dis

    prod = 1
    for i in disrater:
        prod *= int(i)  # numpy overflows here so we need a Python int
    B = len(disrater) * prod

    chanceP = 1 - B / ns**(comb(nr, 2) * 2)
    varkappa = chanceP / (ns * (1 - chanceP))

    SEkappa = np.sqrt(varkappa)
    u = value / SEkappa
    pvalue = 2 * (1 - norm.cdf(np.abs(u)))

    method = "Light's Kappa for m Raters"
    return IRR_result(method, ns, nr, "Kappa", value, u, "z", pvalue)
