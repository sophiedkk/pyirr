import numpy as np
import pandas as pd

from .IRR_result import IRR_result


def iota(ratings, scale_data, standardize=False):
    """Computes iota as an index of interrater agreement of quantitative or nominal multivariate observations.

    Parameters
    ----------
    ratings: list
        list with subjects * raters array or dataframe
    scale_data: {"quantitative", "nominal"}
        a character string specifying if the data is '"quantitative"' (default) or '"nominal"'.
    standardize: bool
       a logical indicating whether quantitative data should be z-standardized within each variable before the
       computation of iota.

    """
    detail = None
    for i, rating in enumerate(ratings):
        rating = pd.DataFrame(rating)
        ratings[i] = rating.dropna()

    ns = ratings[0].shape[0]
    nr = ratings[0].shape[1]

    nvar = len(ratings)  # number of variables

    if scale_data == "quantitative":
        if standardize:
            for i, rating in enumerate(ratings):
                x = np.ravel(rating)
                ratings[i] = (rating - x.mean()) / x.std()
            detail = "Variables have been z-standardized before the computation"
        ratinglist = [np.array(rating) for rating in ratings]  # Take original as new rating-structure
    elif scale_data == "nominal":
        ratinglist = []
        dummyn = 0
        for i, rating in enumerate(ratings):
            # How many levels were used?
            levels = np.unique(rating.values)

            for level in levels:
                # Build new rating-structure
                ratinglist.append(np.zeros((ns, nr)))
                ratinglist[dummyn][ratings[i] == level] = 1 / np.sqrt(2)
                dummyn += 1
    else:
        raise Exception("Please choose quantitative or nominal.")

    # Compute coefficient
    doSS, deSS = 0, 0

    for rating in ratinglist:
        SSt = np.cov(rating.ravel()) * (ns * nr - 1)
        SSw = np.sum(np.apply_along_axis(np.cov, 1, rating) / ns) * ns * (nr - 1)
        SSc = np.cov(np.mean(rating, 0)) * ns * (nr - 1)

        doSS = doSS + SSw
        deSS = deSS + ((nr - 1) * SSt + SSc)

    coeff = 1 - (nr * doSS) / deSS

    return IRR_result(f"iota for {scale_data} ({nvar} variable(s))", ns, nr, "iota", coeff, detail=detail)


# > iota(list(diagnoses))  #TODO: write tests, this one works
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
# diagnoses = pd.read_csv("pyrr/tests/diagnoses.csv", index_col=0)
# iota([diagnoses], "nominal")
#
#
# photo = []
# photo.append(np.array([[71, 74, 76], [73, 80, 80], [86, 101, 93], [59, 62, 66], [71, 83, 77]]))
# photo.append(np.array([[166, 171, 171], [160, 170, 165], [187, 174, 185], [161, 163, 162], [172, 182, 181]]))
# iota(photo, "quantitative", True)
