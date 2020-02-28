from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class kappa2_result:
    subjects: int
    raters: int
    value: float
    weight: str

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + f"Cohen's Kappa for 2 Raters (Weights: {self.weight})".center(50, " ")
        model_string += "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\nA = {self.value:.3f}\n" + "=" * 50
        return model_string


def kappa2(ratings, weight, numeric=True, sort_levels=False):
    """Cohen’s Kappa and weighted Kappa for two raters

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    weight: {"unweighted", "equal", "squared"}
        either a character string specifying one predefined set of weights or a numeric vector with own weights.
    numeric: bool
        whether or not data should be interpreted as numeric or as a factor
    sort_levels: bool
        boolean value describing whether factor levels should be (re-)sorted during the calculation

    """
    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if nr > 2:
        raise Exception("Number of raters exceeds 2. Try kappam_fleiss or kappam_light")

    r1, r2 = ratings[:, 0], ratings[:, 1]

    sort_levels = True if numeric else sort_levels

    lev = set(ratings.ravel())

    nc = len(set(r2))

    if not isinstance(weight, str):
        w = 1 - (weight - min(weight)) / (max(weight) - min(weight))
    elif weight == "equal":
        w = np.arange((nc - 1), -1, -1) / (nc-1)
    elif weight == "squared":
        w = 1 - np.arange(0, nc, 1)**2 / (nc - 1)**2
    elif weight == "unweighted":
        w = np.zeros(nc)
        w[0] = 1
    # TODO: continue





    SS_total = np.cov(np.ravel(ratings)) * (ns * nr - 1)
    SSb = np.cov(ratings.mean(axis=1)) * nr * (ns - 1)
    SSw = np.cov(ratings.mean(axis=0)) * ns * (nr - 1)
    SSr = SS_total - SSb - SSw

    coeff = SSb / (SSb + SSr)

    return kappa2_result(ns, nr, coeff, weight)
