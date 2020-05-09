import pandas as pd

from pkg_resources import resource_filename


def read_data(name):
    """Reads sample data into DataFrame

    Parameters
    ----------
    name: str
        {"anxiety", "diagnoses", "gonio", "nmm", "photo", "video", "vision"}

    """

    if name == "photo":
        return pd.read_pickle(resource_filename("pyrr", "data/photo.p"))
    else:
        return pd.read_csv(resource_filename("pyrr", f"data/{name}.csv"), index_col=0)
