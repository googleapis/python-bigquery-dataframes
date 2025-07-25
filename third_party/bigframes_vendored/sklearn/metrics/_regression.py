"""Metrics to assess performance on regression task.
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Arnaud Joly <a.joly@ulg.ac.be>
#          Jochen Wersdorfer <jochen@wersdoerfer.de>
#          Lars Buitinck
#          Joel Nothman <joel.nothman@gmail.com>
#          Karan Desai <karandesai281196@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Manoj Kumar <manojkumarsivaraj334@gmail.com>
#          Michael Eickenberg <michael.eickenberg@gmail.com>
#          Konstantin Shmelkov <konstantin.shmelkov@polytechnique.edu>
#          Christian Lorentzen <lorentzen.ch@gmail.com>
#          Ashutosh Hathidara <ashutoshhathidara98@gmail.com>
#          Uttam kumar <bajiraouttamsinha@gmail.com>
#          Sylvain Marie <sylvain.marie@se.com>
#          Ohad Michel <ohadmich@gmail.com>
# License: BSD 3 clause

from bigframes import constants


def r2_score(y_true, y_pred, force_finite=True) -> float:
    """:math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    In the particular case when ``y_true`` is constant, the :math:`R^2` score
    is not finite: it is either ``NaN`` (perfect predictions) or ``-Inf``
    (imperfect predictions). To prevent such non-finite numbers to pollute
    higher-level experiments such as a grid search cross-validation, by default
    these cases are replaced with 1.0 (perfect predictions) or 0.0 (imperfect
    predictions) respectively.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.ml.metrics
        >>> bpd.options.display.progress_bar = None

        >>> y_true = bpd.DataFrame([3, -0.5, 2, 7])
        >>> y_pred = bpd.DataFrame([2.5, 0.0, 2, 8])
        >>> r2_score = bigframes.ml.metrics.r2_score(y_true, y_pred)
        >>> r2_score
        np.float64(0.9486081370449679)

    Args:
        y_true (Series or DataFrame of shape (n_samples,)):
            Ground truth (correct) target values.
        y_pred (Series or DataFrame of shape (n_samples,)):
            Estimated target values.

    Returns:
        float: The :math:`R^2` score.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)


def mean_squared_error(y_true, y_pred) -> float:
    """Mean squared error regression loss.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.ml.metrics
        >>> bpd.options.display.progress_bar = None

        >>> y_true = bpd.DataFrame([3, -0.5, 2, 7])
        >>> y_pred = bpd.DataFrame([2.5, 0.0, 2, 8])
        >>> mse = bigframes.ml.metrics.mean_squared_error(y_true, y_pred)
        >>> mse
        np.float64(0.375)

    Args:
        y_true (Series or DataFrame of shape (n_samples,)):
            Ground truth (correct) target values.
        y_pred (Series or DataFrame of shape (n_samples,)):
            Estimated target values.

    Returns:
        float: Mean squared error.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean absolute error regression loss.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.ml.metrics
        >>> bpd.options.display.progress_bar = None

        >>> y_true = bpd.DataFrame([3, -0.5, 2, 7])
        >>> y_pred = bpd.DataFrame([2.5, 0.0, 2, 8])
        >>> mae = bigframes.ml.metrics.mean_absolute_error(y_true, y_pred)
        >>> mae
        np.float64(0.5)

    Args:
        y_true (Series or DataFrame of shape (n_samples,)):
            Ground truth (correct) target values.
        y_pred (Series or DataFrame of shape (n_samples,)):
            Estimated target values.

    Returns:
        float: Mean absolute error.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
