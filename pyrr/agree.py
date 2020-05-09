import numpy as np
import pandas as pd

from .IRR_result import IRR_result


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
    ratings = pd.DataFrame(ratings)  # make sure ratings is not a list or DataFrame

    ratings.dropna(inplace=True)  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    if numeric:
        range_tab = np.max(ratings, axis=1) - np.min(ratings, axis=1)
        coeff = 100 * np.sum(range_tab <= tolerance) / ns
    else:
        range_tab = ratings.apply(lambda row: len(row.value_counts()), axis=1)
        coeff = 100 * (np.sum(range_tab == 1) / ns)
        tolerance = 0

    return IRR_result(f"Percentage agreement (Tolerance={tolerance:.2f})", ns, nr, "%-agree", coeff)
