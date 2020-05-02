from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import f


@dataclass
class finn_result:
    subjects: int
    raters: int
    df2: int
    model: str
    value: float
    statistic: float
    p_value: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + f"Finn-Coefficient (Model={self.model})".center(50, " ")
        model_string += "\n" + "=" * 50 + "\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\nFinn = {self.value:.3f}\n\n"
        model_string += f"F(Inf, {self.df2}): {self.statistic:.1f}\np_value: {self.p_value:.3f}\n"
        model_string += "=" * 50
        return model_string


def finn(ratings, s_levels, model):
    """Computes the Finn coefficient as an index of the interrater reliability of quantitative data.

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    s_levels: float
        the number of different rating categories
    model: bool
        a character string specifying if a '"oneway"' model (default) with row effects random, or a '"twoway"' model
        with column and row effects random should be applied.

    """
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
        p_value = 1 - f.cdf(Fvalue, df1, df2)
    elif model == "twoway":
        coeff = 1 - (MSe / MSexp)
        Fvalue = MSexp / MSe
        p_value = 1 - f.cdf(Fvalue, df1, df2)

    return finn_result(ns, nr, df2, model, coeff, Fvalue, p_value)


# agree(video)  # TODO: test cases, this one is correct
# Finn-Coefficient (Model=twoway)
#
#  Subjects = 20
#    Raters = 4
#      Finn = 0.925
#
# F(Inf,60) = 13.3
#   p-value = 1.74e-23