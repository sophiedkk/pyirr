import numpy as np
import pandas as pd
from scipy.stats import norm

from .IRR_result import IRR_result


def kappam_fleiss(ratings, exact=False, detail=False):
    """Computes Fleiss' Kappa as an index of interrater agreement between m raters on categorical data. Additionally,
    category-wise Kappas could be computed.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    exact: bool
        a boolean indicating whether the exact Kappa (Conger, 1980) or the Kappa described by Fleiss (1971) should be
        computed.
    detail: bool
        a boolean indicating whether category-wise Kappas should be computed

    Returns
    -------
    IRR_result
        Returns Fleiss' Kappa as an IRR_result dataclass.

   """
    ratings = pd.DataFrame(ratings)
    ns = ratings.shape[0]
    nr = ratings.shape[1]

    lev = np.unique(ratings)

    ttab = ratings.apply(lambda row: pd.Categorical(row, lev).value_counts(), axis=1)

    ttab = ttab.values

    agreeP = np.sum((np.sum(ttab**2, axis=1) - nr) / (nr * (nr - 1)) / ns)

    if exact:
        method = "Fleiss` Kappa for m Raters (exact value)"
        rtab = ratings.apply(lambda col: pd.Categorical(col, lev).value_counts(), axis=0)
        rtab /= ns

        cov = np.apply_along_axis(np.cov, 1, rtab)
        chanceP = np.sum(np.sum(ttab, axis=0)**2) / (ns * nr)**2 - np.sum(cov * (nr - 1) / nr) / (nr - 1)
    else:
        method = "Fleiss` Kappa for m Raters"
        chanceP = np.sum(np.sum(ttab, axis=0)**2) / (ns * nr)**2

    value = (agreeP - chanceP) / (1 - chanceP)

    if not exact:
        pj = np.sum(ttab, axis=0) / (ns * nr)
        qj = 1 - pj

        varkappa = (2 / (sum(pj * qj)**2 * (ns * nr * (nr - 1)))) * (sum(pj * qj)**2 - sum(pj * qj * (qj - pj)))
        SEkappa = np.sqrt(varkappa)

        u = value/SEkappa
        pvalue = 2 * (1 - norm.cdf(np.abs(u)))

        if detail:
            pj = np.sum(ttab, axis=0) / (ns * nr)
            pjk = (np.sum(ttab**2, axis=0) - ns * nr * pj) / (ns * nr * (nr - 1) * pj)

            kappaK = (pjk - pj) / (1 - pj)

            varkappaK = 2 / (ns * nr * (nr-1))
            SEkappaK = np.sqrt(varkappaK)

            uK = kappaK / SEkappaK
            p_valueK = 2 * (1 - norm.cdf(np.abs(uK)))

            tableK = pd.DataFrame([kappaK, uK, p_valueK], columns=lev, index=["Kappa", "z", "p.value"])
            tableK = tableK.round(3).T

    rval = {"method": method, "subjects": ns, "raters": nr, "irr_name": "Kappa", "value": value}

    if not exact:
        if detail:
            rval = {**rval, "detail": tableK}

        rval = {**rval, "stat_name": "z", "statistic": u, "pvalue": pvalue}

    return IRR_result(**rval)
