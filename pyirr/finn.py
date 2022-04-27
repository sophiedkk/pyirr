import numpy as np
from scipy.stats import f

from .IRR_result import IRR_result


def finn(ratings, s_levels, model):
    """Computes the Finn coefficient as an index of the interrater reliability of quantitative data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    s_levels: float
        the number of different rating categories
    model: {"oneway", "twoway"}
        a character string specifying if a '"oneway"' model (default) with row effects random, or a '"twoway"' model
        with column and row effects random should be applied.

    Returns
    -------
    IRR_result
        Finn coefficient in an IRR_result dataclass

    """
    if model not in ("oneway", "twoway"):
        raise ValueError("Model should be either 'oneway' or 'twoway'.")

    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    SS_total = np.cov(ratings.ravel()) * (ns * nr - 1)
    MSr = np.cov(np.mean(ratings, axis=1)) * nr
    MSw = np.sum(np.apply_along_axis(np.cov, axis=1, arr=ratings) / ns)
    MSc = np.apply_along_axis(np.cov, axis=0, arr=np.mean(ratings, axis=0)) * ns
    MSe = (SS_total - MSr * (ns - 1) - MSc * (nr - 1)) / ((ns - 1) * (nr - 1))

    MSexp = 1 / 12 * (s_levels**2 - 1)

    df1 = 10e100  # Inf, holistic approach
    df2 = ns * (nr - 1)

    if model == "oneway":
        coeff = 1 - (MSw / MSexp)
        Fvalue = MSexp / MSw
        pvalue = 1 - f.cdf(Fvalue, df1, df2)
    elif model == "twoway":
        coeff = 1 - (MSe / MSexp)
        Fvalue = MSexp / MSe
        pvalue = 1 - f.cdf(Fvalue, df1, df2)

    return IRR_result(f"Finn-Coefficient (Model={model})", ns, nr, "Finn", coeff, Fvalue, f"F(Inf,{df2})", pvalue)
