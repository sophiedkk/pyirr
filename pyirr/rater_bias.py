import numpy as np
import pandas as pd
from scipy.stats import chi2

from .IRR_result import IRR_result


def rater_bias(ratings):
    """Calculates a coefficient of systematic bias between two raters.

    Parameters
    ----------
    ratings: array_like
        n x 2 matrix of classification scores into c categories.

    Returns
    -------
    IRR_result
        Bias as an IRR_result dataclass.
    """
    ratings = pd.DataFrame(ratings)

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("More than two raters, cannot compute")

    rbx = pd.crosstab(ratings.iloc[:, 0], ratings.iloc[:, 1])
    rbb = np.sum(np.triu(rbx, k=len(rbx)//2 - 1))
    rbc = np.sum(np.tril(rbx, k=-len(rbx)//2 + 1))
    rb = np.abs(rbb/(rbb+rbc))
    rbstat = (rbb - rbc)**2 / (rbb + rbc)

    pvalue = 1 - chi2.cdf(rbstat, 1)

    return IRR_result("Rater bias coefficient", ns, nr, "Ratio", rb, rbstat, f"Chisq(1)", pvalue)
