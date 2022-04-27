import os
from setuptools import setup


def read(filename):
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name                    = 'pyirr',
    version                 = '0.84.1.2',
    description             = 'Python implementation of the R package IRR',
    author                  = 'Rick de Klerk',
    author_email            = 'rickdkk@gmail.com',
    url                     = 'https://github.com/rickdkk/pyirr',
    download_url            = 'https://github.com/rickdkk/pyirr',
    packages                = ['pyirr'],
    package_data            = {"pyirr": ['pyirr/data/*']},
    include_package_data    = True,
    long_description        = read("README.rst"),
    license                 = 'GNU GPLv3',
    keywords                = ['statistics'],
    classifiers             = [],
    install_requires        = ["numpy", "scipy", "pandas", "statsmodels"]
)
