from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class robinson_result:
    subjects: int
    raters: int
    value: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + "Robinson's A".center(50, " ") + "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\nA = {self.value:.3f}\n" + "=" * 50
        return model_string


def robinson(ratings):
    """Computes Robinson’s A as an index of the interrater reliability of quantitative data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe

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

    return robinson_result(ns, nr, coeff)

# robinson(anxiety)  # TODO: test cases, this one is correct
#  Robinson's A
#
#  Subjects = 20
#    Raters = 3
#         A = 0.477
