import numpy as np

from .IRR_result import IRR_result


def robinson(ratings):
    """Computes Robinson’s A as an index of the interrater reliability of quantitative data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

    Returns
    -------
    IRR_result
        Returns Robinson's A as an IRR_result dataclass.

    """
    ratings = np.asarray(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    SS_total = np.cov(np.ravel(ratings)) * (ns * nr - 1)
    SSb = np.cov(ratings.mean(axis=1)) * nr * (ns - 1)
    SSw = np.cov(ratings.mean(axis=0)) * ns * (nr - 1)
    SSr = SS_total - SSb - SSw

    coeff = SSb / (SSb + SSr)

    return IRR_result("Robinson's A", ns, nr, "A", coeff)
