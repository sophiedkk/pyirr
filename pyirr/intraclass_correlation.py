import numpy as np
from scipy.stats import f
from dataclasses import dataclass, asdict


@dataclass
class ICC_result:
    subjects: int
    raters: int
    model: str
    mtype: str
    name: str
    value: float
    r0: float
    Fvalue: float
    df1: float
    df2: float
    pvalue: float
    conf_level: float
    lower_bound: float
    upper_bound: float

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        model_string = "=" * 50 + "\n" + "Intraclass Correlation Results".center(50, " ") + "\n" + "=" * 50 + "\n"
        model_string += f"Model: {self.model}\nType: {self.mtype}\n\n"
        model_string += f"Subjects = {self.subjects}\nRaters = {self.raters}\n{self.name} = {self.value:.2f}\n\n"
        model_string += f"F-Test, H0: r0 = {self.r0} ; H1 : r0 > {self.r0}\n"
        model_string += f"F({self.df1:.2f},{self.df2:.2f}) = {self.Fvalue:.2f}, p = {self.pvalue:.4f}\n\n"
        model_string += f"{self.conf_level*100:.0f}%-Confidence Interval for ICC Population Values:\n"
        model_string += f"{self.lower_bound:.3f} < ICC < {self.upper_bound:.3f}\n"
        model_string += "=" * 50
        return model_string


def intraclass_correlation(ratings, model="oneway", mtype="consistency", unit="single", r0=0, conf_level=0.95):
    """Calculate the intraclass correlation for a set of ratings

    Parameters
    ----------
    ratings: array_like
        subjects * raters array or dataframe
    model: {"oneway", "twoway"}
        whether it is a "oneway" or "twoway" model
    mtype: {"consistency", "agreement"}
        whether you want to test for "consistency" or "agreement"
    unit: {"single", "average"}
        unit of the analysis
    r0: int
        specification of the null hypothesis
    conf_level: float
        confidence level

    Returns
    -------
    ICC_result
        The intraclass correlation statistics in an ICC_result dataclass.

    """
    ratings = np.array(ratings)  # make sure ratings is not a list or DataFrame

    alpha = 1 - conf_level
    ratings = ratings[~np.isnan(ratings).any(axis=1)]  # drop nans

    ns = ratings.shape[0]
    nr = ratings.shape[1]

    SS_total = np.cov(np.ravel(ratings)) * (ns * nr - 1)
    MSr = np.cov(np.mean(ratings, axis=1)) * nr
    MSw = np.sum(np.apply_along_axis(np.cov, axis=1, arr=ratings) / ns)
    MSc = np.apply_along_axis(np.cov, axis=0, arr=np.mean(ratings, axis=0)) * ns
    MSe = (SS_total - MSr * (ns - 1) - MSc * (nr - 1)) / ((ns - 1) * (nr - 1))

    if unit == "single":
        if model == "oneway":  # Assendorpf & Wallbot, S. 245, ICu, Bartko (1966) [3]
            name = "ICC(1)"
            coeff = (MSr - MSw) / (MSr + (nr - 1) * MSw)
            Fvalue = MSr / MSw * ((1 - r0) / (1 + (nr - 1) * r0))
            df1 = ns - 1
            df2 = ns * (nr - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # confindence interval
            FL = (MSr / MSw) / f.ppf(1 - alpha/2, ns - 1, ns * (nr - 1))
            FU = (MSr / MSw) * f.ppf(1 - alpha/2, ns * (nr - 1), ns - 1)
            lbound = (FL - 1) / (FL + nr - 1)
            ubound = (FU - 1) / (FU + nr - 1)
        elif model == "twoway":
            if mtype == "consistency":
                # Asendorpf & Wallbott, S. 245, ICa
                # Bartko (1966), [21]
                # Shrout & Fleiss (1979), ICC(3,1)
                name = "ICC(C,1)"
                coeff = (MSr - MSe) / (MSr + (nr - 1) * MSe)
                Fvalue = MSr / MSe * ((1 - r0) / (1 + (nr - 1) * r0))
                df1 = ns - 1
                df2 = (ns - 1) * (nr - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confindence interval
                FL = (MSr / MSe) / f.ppf(1 - alpha / 2, ns - 1, (ns - 1) * (nr - 1))
                FU = (MSr / MSe) * f.ppf(1 - alpha / 2, (ns - 1) * (nr - 1), ns - 1)
                lbound = (FL - 1) / (FL + (nr - 1))
                ubound = (FU - 1) / (FU + (nr - 1))
            elif mtype == "agreement":
                # Asendorpf & Wallbott, S. 246, ICa'
                # Bartko (1966), [15]
                # Shrout & Fleiss (1979), ICC(2,1)
                name = "ICC(A,1)"
                coeff = (MSr - MSe) / (MSr + (nr - 1) * MSe + (nr / ns) * (MSc - MSe))
                a = (nr * r0) / (ns * (1 - r0))
                b = 1 + (nr * r0 * (ns - 1)) / (ns * (1 - r0))
                Fvalue = MSr / (a * MSc + b * MSe)
                a = (nr * coeff) / (ns * (1 - coeff))
                b = 1 + (nr * coeff * (ns - 1)) / (ns * (1 - coeff))
                v = (a * MSc + b * MSe)**2 / ((a * MSc)**2 / (nr - 1) + (b * MSe)**2 / ((ns - 1) * (nr - 1)))
                df1 = ns - 1
                df2 = v
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval (McGraw & Wong, 1996)
                FL = f.ppf(1 - alpha / 2, ns - 1, v)
                FU = f.ppf(1 - alpha / 2, v, ns - 1)
                lbound = (ns * (MSr - FL * MSe)) / (FL * (nr * MSc + (nr * ns - nr - ns) * MSe) + ns * MSr)
                ubound = (ns * (FU * MSr - MSe)) / (nr * MSc + (nr * ns - nr - ns) * MSe + ns * FU * MSr)
    elif unit == "average":
        if model == "oneway":
            # Asendorpf & Wallbott, S. 245, Ru
            name = f"ICC({nr})"
            coeff = (MSr - MSw) / MSr
            Fvalue = MSr / MSw * (1 - r0)
            df1 = ns - 1
            df2 = ns * (nr - 1)
            pvalue = 1 - f.cdf(Fvalue, df1, df2)

            # confidence interval
            FL = (MSr / MSw) / f.ppf(1 - alpha / 2, ns - 1, ns * (nr - 1))
            FU = (MSr / MSw) * f.ppf(1 - alpha / 2, ns * (nr - 1), ns - 1)
            lbound = 1 - 1 / FL
            ubound = 1 - 1 / FU
        elif model == "twoway":
            if mtype == "consistency":
                # Asendorpf & Wallbott, S. 246, Ra
                name = f"ICC(C,{nr})"
                coeff = (MSr - MSe) / MSr
                Fvalue = MSr / MSe * (1 - r0)
                df1 = ns - 1
                df2 = (ns - 1) * (nr - 1)
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval
                FL = (MSr / MSe) / f.ppf(1 - alpha / 2, ns - 1, (ns - 1) * (nr - 1))
                FU = (MSr / MSe) * f.ppf(1 - alpha / 2, (ns - 1) * (nr - 1), ns - 1)
                lbound = 1 - 1 / FL
                ubound = 1 - 1 / FU
            elif mtype == "agreement":
                name = f"ICC(A,{nr})"
                coeff = (MSr - MSe) / (MSr + (MSc - MSe) / ns)
                a = r0 / (ns * (1 - r0))
                b = 1 + (r0 * (ns - 1)) / (ns * (1 - r0))
                v = (a * MSc + b * MSe)**2 / ((a * MSc)**2 / (nr - 1) + (b * MSe)**2 / ((ns - 1) * (nr - 1)))
                Fvalue = MSr / (a * MSc + b * MSe)
                df1 = ns - 1
                df2 = v
                pvalue = 1 - f.cdf(Fvalue, df1, df2)

                # confidence interval (McGraw & Wong, 1996)
                FL = f.ppf(1 - alpha / 2, ns - 1, v)
                FU = f.ppf(1 - alpha / 2, v, ns - 1)
                lbound = (ns * (MSr - FL * MSe)) / (FL * (MSc - MSe) + ns * MSr)
                ubound = (ns * (FU * MSr - MSe)) / (MSc - MSe + ns * FU * MSr)

    return ICC_result(ns, nr, model, mtype, name, coeff, r0, Fvalue, df1, df2, pvalue, conf_level, lbound, ubound)
