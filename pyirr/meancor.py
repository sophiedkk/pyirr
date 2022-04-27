import numpy as np
from scipy.stats import norm, pearsonr

from .IRR_result import IRR_result


def meancor(ratings, fisher=True):
    """Computes the mean of bivariate Pearson's product moment correlations between raters as an index of the interrater
    reliability of quantitative data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    fisher: bool
        a boolean indicating whether the correlation coefficients should be Fisher z-standardized before averaging.

    Returns
    -------
    IRR_result
        Returns correlation as an IRR_result dataclass.

    """
    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    r = []
    for i in range(nr-1):
        for j in range(i + 1, nr):
            r.append(pearsonr(ratings[:, i], ratings[:, j])[0])

    r = np.array(r)
    delr = 0

    if fisher:
        delr = len(r) - len(r[(r < 1) & (r > -1)])
        # Eliminate perfect correlations (r=1, r=-1)
        r = r[(r < 1) & (r > -1)]

        rz = 0.5 * np.log((1 + r) / (1 - r))
        mrz = np.mean(rz)

        coeff = (np.exp(2 * mrz) - 1) / (np.exp(2 * mrz) + 1)
        SE = np.sqrt(1 / (ns - 3))

        u = coeff / SE
        pvalue = 2 * (1 - norm.cdf(np.abs(u)))

    else:
        coeff = np.mean(r)

    result = {"method": "Mean of bivariate correlations R", "subjects": ns, "raters": nr, "irr_name": "R",
              "value": coeff}
    if fisher:
        result = {**result, "statistic": u, "stat_name": "z", "pvalue": pvalue}
    if delr > 0:
        error = "perfect correlation was" if delr == 1 else "perfect correlations were"
        result = {**result, "error": error + "dropped before averaging"}

    return IRR_result(**result)
