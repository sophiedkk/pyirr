from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .IRR_result import IRR_result


@dataclass
class bhapkar_result:
    subjects: int
    raters: int
    value: float
    pvalue: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + f"Bhapkar marginal homogeneity".center(50, " ")
        model_string += "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\nChisq = {self.value:.0f}\n"
        model_string += f"p-value = {self.pvalue:.5f}\n"
        model_string += "=" * 50
        return model_string


def bhapkar(ratings):
    """Calculate the percentage agreement among raters

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

    """
    ratings = pd.DataFrame(ratings)  # make sure ratings is a DataFrame

    ratings.dropna(axis=0, inplace=True)

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("Number of raters exceeds 2. Try kappam_fleiss or kappam_light")

    r1, r2 = ratings.iloc[:, 0], ratings.iloc[:, 1]

    # find factor levels
    levels = np.unique(ratings)

    r1 = pd.Categorical(r1, categories=levels)
    r2 = pd.Categorical(r2, categories=levels)

    # compute table
    ttab = pd.crosstab(r1, r2, dropna=False)

    # get marginals
    row_sums = ttab.sum(axis=1)[:-1]
    col_sums = ttab.sum(axis=0)[:-1]

    # compute d matrix
    dmat = np.tile(row_sums - col_sums, [len(row_sums), 1])

    # setup delta matrix
    delta = np.zeros((len(row_sums), len(row_sums)))
    np.fill_diagonal(delta, row_sums + col_sums)

    # dump last category from smx table
    smx = ttab.iloc[:-1, :-1]

    # compute w matrix
    w = delta - smx - smx.T - (dmat * dmat.T) / ns
    w1 = np.linalg.inv(w)

    # compute chisq-value
    chimat = dmat * dmat.T * w1

    # test statistics
    x_value = np.sum(chimat)
    df1 = len(ttab) - 1
    pvalue = 1 - chi2.cdf(x_value, df1)

    method = "Bhapkar marginal homogeneity"
    stat_name = f"Chisq({df1})"
    return IRR_result(method, ns, nr, "Chisq", x_value, x_value, stat_name, pvalue)

# bhapkar(vision)  # TODO: add tests, this one is correct
#  Bhapkar marginal homogeneity
#
#  Subjects = 7477
#    Raters = 2
#     Chisq = 12
#
#  Chisq(3) = 12
#   p-value = 0.00747

