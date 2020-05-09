Coefficients of Interrater Reliability and Agreement
====================================================

Python implementation of the R package `IRR <https://CRAN.R-project.org/package=irr>`_.
For documentation please head over to the R package page, they put in a lot of effort for the documentation!

How to install
--------------
Just run::

    pip install pyrr

Example
-------
A simple example::

    from pyrr import read_data, intraclass_correlation

    data = read_data("anxiety")
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
