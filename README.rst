Coefficients of Interrater Reliability and Agreement
====================================================

.. image:: https://github.com/rickdkk/pyirr/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/rickdkk/pyirr/actions

.. image:: https://zenodo.org/badge/484431981.svg
   :target: https://zenodo.org/badge/latestdoi/484431981

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
    :target: https://github.com/rickdkk/pyirr/blob/main/LICENSE

Python implementation of the R package `IRR <https://CRAN.R-project.org/package=irr>`_, all credit goes to the original
authors [1]_. The package contains functions to calculate coefficients of Interrater Reliability and Agreement for interval,
ordinal and nominal data: intraclass-correlations, Finn-Coefficient, Robinson's A, Kendall's W, Cohen's Kappa, and others.
This is a straight line-for-line port from the R-package, so it is not particularly Pythonic and mainly made as an
exercise to learn more about R. For documentation I highly recommend you head over to the R package page, they put in a
lot of effort for the documentation!


How to install
--------------
The package is available on the Python Package Index (PyPI). To install it you can run::

    pip install pyirr

How to use
----------
A simple example::

    from pyirr import read_data, intraclass_correlation

    data = read_data("anxiety")  # loads example data
    intraclass_correlation(data, "twoway", "agreement")

Returns::

    ==================================================
              Intraclass Correlation Results
    ==================================================
    Model: twoway
    Type: agreement

    Subjects = 20
    Raters = 3
    ICC(A,1) = 0.20

    F-Test, H0: r0 = 0 ; H1 : r0 > 0
    F(19.00,39.75) = 1.83, p = 0.0543

    95%-Confidence Interval for ICC Population Values:
    -0.039 < ICC < 0.494
    ==================================================

Another simple example::

    from pyirr import read_data, kappam_fleiss

    data = read_data("anxiety")  # loads example data
    kappam_fleiss(data, detail=True)

Returns::

    ==================================================
                Fleiss` Kappa for m Raters
    ==================================================
    Subjects = 30
      Raters = 6
       Kappa = 0.430

           z = 17.652
     p-value = 0.000

                             Kappa       z  p.value
    1. Depression            0.245   5.192      0.0
    2. Personality Disorder  0.245   5.192      0.0
    3. Schizophrenia         0.520  11.031      0.0
    4. Neurosis              0.471   9.994      0.0
    5. Other                 0.566  12.009      0.0
    ==================================================

.. [1] Gamer, M., Lemon, J., Gamer, M.M., Robinson, A. and Kendall’s, W., 2012. Package ‘irr’. Various coefficients of interrater reliability and agreement, 22.
