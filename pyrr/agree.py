from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class agree_result:
    subjects: int
    raters: int
    value: float
    tolerance: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + f"Percentage agreement (Tolerance={self.tolerance:.2f})".center(50, " ")
        model_string += "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\n%-agree = {self.value:.1f}\n"
        model_string += "=" * 50
        return model_string


def agree(ratings, tolerance=0, numeric=True):
    """Calculate the percentage agreement among raters

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    tolerance: float
        tolerance in ratings before raters differ in agreement
    numeric: bool
        should data be treated as numeric or as categories

    """
    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if numeric:
        range_tab = np.max(ratings, axis=1) - np.min(ratings, axis=1)
        coeff = 100 * np.sum(range_tab <= tolerance) / ns
    else:
        range_tab = np.unique(ratings, axis=1)
        coeff = 100 * (np.sum(range_tab == 1) / ns)
        tolerance = 0

    return agree_result(ns, nr, value=coeff, tolerance=tolerance)
