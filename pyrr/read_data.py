import os
import pandas as pd


def read_data(name):
    """Reads sample data into DataFrame

    Parameters
    ----------
    name: str
        {"anxiety", "diagnoses", "gonio", "nmm", "photo", "video", "vision"}

    """

    if name == "photo":
        return pd.read_pickle(os.path.join(os.path.dirname(__file__), "tests", "photo.p"))
    else:
        return pd.read_csv(os.path.join(os.path.dirname(__file__), "tests", f"{name}.csv"), index_col=0)
