Fetching the Goodness of Fit
============================


The Goodness of Fit is evaluated for the target source region using
:class:`~asgardpy.stats.stats.get_ts_target` and
:class:`~asgardpy.stats.stats.get_goodness_of_fit_stats`.

The different Fit Statistic function used for the different types of Datasets
input are mentioned in the
`Gammapy documentation <https://docs.gammapy.org/1.1/user-guide/datasets/index.html#types-of-supported-datasets>`_
and are:

#. `cash <https://docs.gammapy.org/1.1/api/gammapy.stats.cash.html>`_
    Used for dataset containing Poisson data with background model.

#. `wstat <https://docs.gammapy.org/1.1/api/gammapy.stats.wstat.html>`_
    Used for dataset containing Poisson data with background measurement.

#. chi2
    Used for `FluxPointsDataset` read from a file like in a Gammapy
    `example <https://docs.gammapy.org/1.1/tutorials/analysis-3d/analysis_mwl.html#hawc-1d-dataset-for-flux-point-fitting>`_
    where the pre-computed flux points are used to perform the likelihood fit,
    when no convolution with IRFs are needed.

:class:`~asgardpy.stats.stats.get_ts_target` uses the above Fit Statistic functions,
to get the test statistic for the best fit and the max fit for the target source
region in the provided joint datasets object.

For more general information follow the Gammapy
`documentation <https://docs.gammapy.org/1.1/user-guide/stats/index.html>`_
