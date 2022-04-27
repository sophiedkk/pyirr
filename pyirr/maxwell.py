import numpy as np

from .IRR_result import IRR_result


def maxwell(ratings):
    """Calculate Maxwell's RE

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

    Returns
    -------
    IRR_result
        Returns Maxwell's RE as an IRR_result dataclass.

    """
    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("Number of raters exceeds 2.")

    levels = set(ratings.ravel())  # number of unique levels

    if len(levels) > 2:
        raise Exception("Ratings are not binary.")

    r1, r2 = ratings[:, 0], ratings[:, 1]

    coeff = 2 * np.sum((r1 - r2) == 0) / ns - 1

    return IRR_result("Maxwell's RE", ns, nr, "RE", coeff)
