import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
	name             		= 'pyirr',
	version          		= '0.84.1.1',
	description      		= 'Python implementation of the R package IRR',
	author           		= 'Rick de Klerk',
	author_email     		= 'r.de.klerk@umcg.nl',
	url              		= 'https://gitlab.com/Rickdkk/pyrr',
	download_url     		= 'https://gitlab.com/Rickdkk/pyrr',
	packages         		= ['pyirr'],
	package_data     		= {"pyirr": ['pyirr/data/*']},
	include_package_data 	= True,
	long_description 		= read("README.rst"),
	license 				= 'GNU GPLv3',
	keywords         		= ['statistics'],
	classifiers      		= [],
	install_requires 		= ["numpy", "scipy", "pandas", "statsmodels"]
)
