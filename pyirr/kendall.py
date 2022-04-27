import numpy as np
import pandas as pd
from scipy.stats import chi2

from .IRR_result import IRR_result


def kendall(ratings, correct=False):
    """Computes Kendall's coefficient of concordance as an index of interrater reliability of ordinal data. The
    coefficient could be corrected for ties within raters.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    correct: bool
        a logical indicating whether the coefficient should be corrected for ties within raters.

    Returns
    -------
    IRR_result
        Returns Kendall's coefficient of concordance as an IRR_result dataclass.

   """
    ratings = pd.DataFrame(ratings)
    ns = ratings.shape[0]
    nr = ratings.shape[1]

    ratings.dropna(inplace=True)  # drop nans

    if correct:
        ratings_rank = ratings.rank()
        error = None

        Tj = 0
        for col in ratings_rank:
            rater = ratings_rank[col].value_counts()
            ties = rater[rater > 1]
            Tj += np.sum(ties**3 - ties)

        coeff = (12 * np.cov(np.sum(ratings_rank, axis=1)) * (ns - 1)) / (nr**2 * (ns**3 - ns) - nr * Tj)
    else:
        testties = np.unique(ratings, axis=0)

        if not len(testties):
            error = "Coefficient may be incorrect due to ties"
        else:
            if len(testties) < len(ratings):
                error = "Coefficient may be incorrect due to ties"

        ratings_rank = ratings.rank()

        coeff = (12 * np.cov(np.sum(ratings_rank, axis=1)) * (ns - 1) / (nr**2 * (ns**3 - ns)))

    Xvalue = nr * (ns - 1) * coeff
    df1 = ns - 1
    pvalue = 1 - chi2.cdf(Xvalue, df1)

    method = "Kendall's coefficient of concordance"
    return IRR_result(method, ns, nr, "Wt", coeff, Xvalue, f"Chisq ({df1})", pvalue, error=error)
