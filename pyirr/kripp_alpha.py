import numpy as np

from .IRR_result import IRR_result


def coincidence_matrix(array):
    array = np.asarray(array)
    levx = np.unique(array)
    levx = levx[~np.isnan(levx)]
    nval = len(levx)
    cm = np.zeros((nval, nval))
    dimx = array.shape

    if np.any(np.isnan(array)):
        mc = np.sum(~np.isnan(array), axis=0) - 1
    else:
        mc = np.ones(dimx[1])

    for col in range(dimx[1]):  # not very Pythonic I know, I'm sorry Guido
        for i1 in range(dimx[0] - 1):
            for i2 in range(i1 + 1, dimx[0]):
                if not np.isnan(array[i1, col]) and not np.isnan(array[i2, col]):
                    index1 = np.nonzero(levx == array[i1, col])
                    index2 = np.nonzero(levx == array[i2, col])
                    cm[index1, index2] += (1 + (index1 == index2)) / mc[col]
                    if index1 != index2:
                        cm[index2, index1] = cm[index1, index2]
    nmv = np.sum(np.sum(cm, axis=0))
    return cm, nmv, levx


def kripp_alpha(ratings, method="nominal"):
    """Calculates the alpha coefficient of reliability proposed by Krippendorff

    Parameters
    ----------
    ratings: array_like
        observation x rater array or dataframe
    method: {"nominal", "ordinal", "interval", "ratio"}
        data level of x

    Returns
    -------
    IRR_result
        Returns Krippendorff's coefficient as an IRR_result dataclass.

   """
    ratings = np.array(ratings).T
    cm, nmv, levx = coincidence_matrix(ratings)
    dimcm = cm.shape

    triu1, triu2 = np.triu_indices_from(cm, k=len(cm)//2 - 1)
    triu2inds = triu2.argsort()  # R sorts rows first, numpy columns first
    triu1 = triu1[triu2inds]
    triu2 = triu2[triu2inds]
    utcm = cm[triu1, triu2]

    nc = np.sum(cm, axis=1)
    dv = levx

    diff2 = np.zeros(len(utcm))
    ncnk = np.zeros(len(utcm))
    ck = 0

    if dimcm[1] < 2:
        value = 1.
    else:
        for k in range(1, dimcm[1]):
            for c in range(k):
                ncnk[ck] = nc[c] * nc[k]
                if method == "nominal":
                    diff2[ck] = 1.
                elif method == "ordinal":
                    diff2[ck] = nc[c] / 2
                    if k > c:
                        for g in range(c+1, k):
                            diff2[ck] += nc[g]
                        diff2[ck] += nc[k] / 2
                        diff2[ck] **= 2
                elif method == "interval":
                    diff2[ck] = (dv[c] - dv[k])**2
                elif method == "ratio":
                    diff2[ck] = (dv[c] - dv[k])**2 / (dv[c] + dv[k])**2
                else:
                    raise Exception("Please specify a nominal/ordinal/interval/ratio data level")

                ck += 1

        value = 1 - (nmv - 1) * np.sum(utcm * diff2) / np.sum(ncnk * diff2)

    return IRR_result(f"Krippendorff's alpha ({method})", ratings.shape[1], ratings.shape[0], "alpha", value)
