import numpy as np
from scipy.stats import norm, spearmanr

from .IRR_result import IRR_result


def meanrho(ratings, fisher=True):
    """Computes the mean of bivariate Spearman's rho rank correlations between raters as an index of the interrater
    reliability of ordinal data.

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

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    testties = np.unique(ratings, axis=0)

    ties = False
    if len(testties):
        ties = True
    elif len(testties) < len(ratings):
        ties = True

    r = []
    for i in range(nr-1):
        for j in range(i + 1, nr):
            r.append(spearmanr(ratings[:, i], ratings[:, j])[0])

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

    result = {"method": "Mean of bivariate correlations Rho", "subjects": ns, "raters": nr, "irr_name": "Rho",
              "value": coeff}
    if fisher:
        result = {**result, "statistic": u, "stat_name": "z", "pvalue": pvalue}
    if delr > 0:
        error = "perfect correlation was" if delr == 1 else "perfect correlations were"
        result = {**result, "error": error + "dropped before averaging"}
        if ties:
            result["error"] += "\nCoefficient may be incorrect due to ties"

    if delr == 0 and ties:
        result["error"] = "Coefficient may be incorrect due to ties"

    return IRR_result(**result)
