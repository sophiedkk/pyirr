
from setuptools import setup


setup(
	name             		= 'pyrr',
	version          		= '0.84.1',
	description      		= 'Python implementation of the R package IRR',
	author           		= 'Rick de Klerk',
	author_email     		= 'r.de.klerk@umcg.nl',
	url              		= 'https://gitlab.com/Rickdkk/pyrr',
	download_url     		= 'https://gitlab.com/Rickdkk/pyrr',
	packages         		= ['pyrr'],
	package_data     		= {},
	include_package_data 	= True,
	long_description 		= '..',
	license 				= 'GNU GPLv3',
	keywords         		= ['statistics'],
	classifiers      		= [],
	install_requires 		= ["numpy", "scipy", "pandas"]
)
