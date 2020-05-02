from dataclasses import dataclass, asdict

import numpy as np
from scipy.stats import f


@dataclass
class iota_result:
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


def iota(ratings, scale_data, standardize=False):
    """Computes iota as an index of interrater agreement of quantitative or nominal multivariate observations.

    Parameters
    ----------
    ratings: list
        list with subjects * raters array or dataframe
    scale_data: {"quantitative", "nominal"}
        a character string specifying if the data is '"quantitative"' (default) or '"nominal"'. If the data is organized
        in factors, '"nominal"' is chosen automatically.
    standardize: bool
       a logical indicating whether quantitative data should be z-standardized within each variable before the
       computation of iota.

    """

    for i, rating in enumerate(ratings):
        rating = np.asarray(rating)
        ratings[i] = rating[~np.isnan(rating).any(axis=1)]  # drop nans

    ns = ratings[0].shape[0]
    nr = ratings[0].shape[1]

    nvar = len(ratings)  # number of variables

    if scale_data == "quantitative":
        if standardize:
            for i, rating in enumerate(ratings):
                ratings[i] = (rating - rating.mean()) / rating.std()
            detail = "Variables have been z-standardized before the computation"
        ratinglist = ratings  # Take original as new rating-structure
    elif scale_data == "nominal":
        dummyn = 1  # Number of dimensions in dummy list
        ratinglist = []

        levels = set()
        for i, rating in enumerate(ratings):
            # How many levels were used?
            levels |= set(rating)

        # Build new rating-structure


# > iota(list(diagnoses))
 # iota for nominal data (1 variable)
 #
 # Subjects = 30
 #   Raters = 6
 #     iota = 0.442


# > photo <- list()
# > photo[[1]] <- cbind(c( 71, 73, 86, 59, 71),  # weight ratings
# +                     c( 74, 80,101, 62, 83),
# +                     c( 76, 80, 93, 66, 77))
# > photo[[2]] <- cbind(c(166,160,187,161,172),  # height rating
# +                     c(171,170,174,163,182),
# +                     c(171,165,185,162,181))
# > iota(photo)
#  iota for quantitative data (2 variables)
#
#  Subjects = 5
#    Raters = 3
#      iota = 0.755
# > iota(photo, standardize=TRUE) # iota over standardized values
#  iota for quantitative data (2 variables)
#
#  Subjects = 5
#    Raters = 3
#      iota = 0.745
#
# Variables have been z-standardized before the computation

