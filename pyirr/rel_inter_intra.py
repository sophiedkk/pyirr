from pprint import pformat

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from statsmodels.formula.api import ols

from .IRR_result import IRR_result


def rel_inter_intra(ratings, nraters, rho_inter=0.6, rho_intra=0.8, conf_level=0.95):
    """Calculates inter- and intra-rater reliability coefficients.

    Parameters
    ----------
    ratings: array_like
        dataframe or array of rater by object scores with consecutive measurements for each rater in adjacent columns
    nraters: int
        number of raters
    rho_inter: float
        null hypothesis value for the inter-rater reliability coefficient
    rho_intra: float
        null hypothesis value for the intra-rater reliability coefficient
    conf_level: float
        confidence level for the one-sided confidence interval reported

    """
    ratings = pd.DataFrame(ratings)

    ns = ratings.shape[0]
    nmeas = ratings.shape[1] // nraters

    data = np.vstack([np.tile(np.arange(1, ns + 1), int(nraters * nmeas)),
                      np.repeat(np.arange(1, nraters + 1), ns * nmeas),
                      np.tile(np.repeat(np.arange(1, nmeas + 1), ns), nraters),
                      np.ravel(ratings, order="F")]).T
    frame1 = pd.DataFrame(data, columns=["Subject", "Rater", "Repetition", "Result"])
    frame1["Subject"] = frame1["Subject"].astype("category")
    frame1["Rater"] = frame1["Rater"].astype("category")
    frame1["Repetition"] = frame1["Repetition"].astype("category")

    nn = ns  # aliases for compatibility with Eliasziw et al. notation
    tt = nraters
    mm = nmeas

    mean_squares = []
    aov = ols('Result ~ Subject * Rater', data=frame1).fit()
    aov_table = sm.stats.anova_lm(aov, typ=2)
    mean_squares.append(aov_table["sum_sq"] / aov_table["df"])

    for rater_act in frame1["Rater"].cat.categories:
        aov_act = ols('Result ~ Subject', data=frame1[frame1["Rater"] == rater_act]).fit()
        aov_act_table = sm.stats.anova_lm(aov_act, typ=2)
        mean_squares.append((aov_act_table["sum_sq"] / aov_act_table["df"]).iloc[-1])

    MSS = mean_squares[0][0]
    MSR = mean_squares[0][1]
    MSSR = mean_squares[0][2]
    MSE = mean_squares[0][3]

    # the same for random and fixed, see table 2 (p. 780) and 3 (p.281)
    MSEpart = np.array(mean_squares[1:])
    sighat2Srandom = (MSS - MSSR) / (mm * tt)
    sighat2Rrandom = (MSR - MSSR) / (mm * nn)
    sighat2SRrandom = (MSSR - MSE) / mm

    # the same for random and fixed, see table 2 (p. 780) and 3 (p. 281)
    sighat2e = MSE
    sighat2Sfixed = (MSS - MSE) / (mm * tt)
    # sighat2Rfixed = (MSR - MSSR) / (mm * nn)  # not used
    sighat2SRfixed = (MSSR - MSE) / mm

    # the same for random and fixed, see table 2 (p. 780) and 3 (p. 281)
    sighat2e_part = MSEpart
    rhohat_inter_random = sighat2Srandom / (sighat2Srandom + sighat2Rrandom + sighat2SRrandom + sighat2e)
    rhohat_inter_fixed = (sighat2Sfixed - sighat2SRfixed / tt) / (sighat2Sfixed + (tt - 1) *
                                                                  sighat2SRfixed / tt + sighat2e)
    rhohat_intra_random = (sighat2Srandom + sighat2Rrandom + sighat2SRrandom) / (sighat2Srandom + sighat2Rrandom +
                                                                                 sighat2SRrandom + sighat2e)
    rhohat_intra_fixed = (sighat2Sfixed + (tt - 1) * sighat2SRfixed / tt) / (sighat2Sfixed + (tt - 1) *
                                                                             sighat2SRfixed / tt + sighat2e)
    rhohat_intra_random_part = (sighat2Srandom + sighat2Rrandom + sighat2SRrandom) / (sighat2Srandom + sighat2Rrandom +
                                                                                      sighat2SRrandom + sighat2e_part)
    rhohat_intra_fixed_part = (sighat2Sfixed + (tt - 1) * sighat2SRfixed / tt) / (sighat2Sfixed + (tt - 1) *
                                                                                  sighat2SRfixed / tt + sighat2e_part)

    F_inter = (1 - rho_inter) * MSS / ((1 + (tt - 1) * rho_inter) * MSSR)
    F_inter_p = 1 - f.cdf(F_inter, nn - 1, (nn - 1) * (tt - 1))
    alpha = 1 - conf_level

    nu1 = (nn - 1) * (tt - 1) * (tt * rhohat_inter_random * (MSR - MSSR) + nn * (1 + (tt - 1) * rhohat_inter_random) *
          MSSR + nn * tt * (mm - 1) * rhohat_inter_random * MSE) ** 2 / ((nn - 1) * (tt * rhohat_inter_random) ** 2 *
          MSR ** 2 + (nn * (1 + (tt - 1) * rhohat_inter_random) - tt * rhohat_inter_random) ** 2 * MSSR ** 2 +
          (nn - 1) * (tt - 1) * (nn * tt * (mm - 1)) * rhohat_inter_random ** 2 * MSE ** 2)
    nu2 = (nn - 1) * (tt - 1) * (nn * (1 + (tt - 1) * rhohat_inter_fixed) * MSSR + nn * tt * (mm - 1) *
           rhohat_inter_fixed * MSE) ** 2 / ((nn * (1 + (tt - 1) * rhohat_inter_fixed)) ** 2 * MSSR ** 2 + (nn - 1) *
          (tt - 1) * (nn * tt * (mm - 1)) * rhohat_inter_fixed ** 2 * MSE ** 2)

    F1 = f.ppf(1 - alpha, nn - 1, nu1)
    F2 = f.ppf(1 - alpha, nn - 1, nu2)

    lowinter_random = nn * (MSS - F1 * MSSR) / (nn * MSS + F1 * (tt * (MSR - MSSR) + nn * (tt - 1) *
                                                                 MSSR + nn * tt * (mm - 1) * MSE))
    lowinter_random = min(lowinter_random, 1)

    lowinter_fixed = nn * (MSS - F2 * MSSR) / (nn * MSS + F2 * (nn * (tt - 1) * MSSR + nn * tt * (mm - 1) * MSE))
    lowinter_fixed = min(lowinter_fixed, 1)

    F_intra = (1 - rho_intra) * MSS / ((1 + (mm - 1) * rho_intra) * MSE * tt)
    F_intra_p = 1 - f.cdf(F_intra, nn - 1, nn * (mm - 1))

    F_intra_part = (1 - rho_intra) * MSS / ((1 + (mm - 1) * rho_intra) * sighat2e_part * tt)
    F_intra_part_p = 1 - f.cdf(F_intra_part, nn - 1, nn * (mm - 1))

    F3 = 1 - f.ppf(1 - alpha, nn - 1, nn * (mm - 1))

    low_intra = (MSS / tt - F3 * MSE) / (MSS / tt + F3 * (mm - 1) * MSE)
    low_intra = min(low_intra, 1)

    F4 = f.ppf(1 - alpha, nn - 1, nn * (mm - 1))
    low_intra_part = (MSS / tt - F4 * MSEpart) / (MSS / tt + F4 * (mm - 1) * MSEpart)

    low_intra_part = np.clip(low_intra_part, 1, None)

    SEMintra = np.sqrt(MSE)
    SEMintra_part = np.sqrt(MSEpart)
    SEMinter_random = np.sqrt(sighat2Rrandom + sighat2SRrandom + sighat2e)
    SEMinter_fixed = np.sqrt(sighat2SRfixed + sighat2e)

    detail = {"rohat": {"rhohat_inter_random": rhohat_inter_random, "rhohat_intra_random": rhohat_intra_random,
                        "rhohat_inter_fixed": rhohat_inter_fixed, "rhohat_intra_fixed": rhohat_intra_fixed,
                        "rhohat_intra_random_part": rhohat_intra_random_part,
                        "rhohat_intra_fixed_part": rhohat_intra_fixed_part},
              "Fs": {"F_inter": F_inter, "F_intra": F_intra, "F_intra_part": F_intra_part},
              "pvalue": {"F_inter_p": F_inter_p, "F_intra_p": F_intra_p, "F_intra_part_p": F_intra_part_p},
              "lowvalue": {"lowinter_random": lowinter_random, "lowinter_fixed": lowinter_fixed,
                           "low_intra": low_intra, "low_intra_part": low_intra_part},
              "sem ": {"SEMintra": SEMintra, "SEMintra_part": SEMintra_part, "SEMinter_random": SEMinter_random,
                       "SEMinter_fixed": SEMinter_fixed}}

    for key, value in detail.items():  # do some rounding in the nested dicts
        detail[key] = {nkey: np.round(nvalue, 4) for nkey, nvalue in value.items()}

    return IRR_result("Inter/Intrarater reliability", ns, nraters, "rhohat", np.NaN, detail=pformat(detail))
