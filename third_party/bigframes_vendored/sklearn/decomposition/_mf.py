""" Matrix Factorization.
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <denis-alexander.engemann@inria.fr>
#         Michael Eickenberg <michael.eickenberg@inria.fr>
#         Giorgio Patrini <giorgio.patrini@anu.edu.au>
#
# License: BSD 3 clause

from abc import ABCMeta

from bigframes_vendored.sklearn.base import BaseEstimator

from bigframes import constants


class MatrixFactorization(BaseEstimator, metaclass=ABCMeta):
    """Matrix Factorization (MF).

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> from bigframes.ml.decomposition import MatrixFactorization
        >>> X = bpd.DataFrame([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
        >>> model = MatrixFactorization(n_components=2, init='random', random_state=0)
        >>> W = model.fit_transform(X)
        >>> H = model.components_

    Args:
        num_factors (int or auto, default auto):
            Specifies the number of latent factors to use.
            If you aren't running hyperparameter tuning, then you can specify an INT64 value between 2 and 200. The default value is log2(n), where n is the number of training examples.
        user_col (str):
            The user column name.
        item_col (str):
            The item column name.
        l2_reg (float, default 1.0):
            If you aren't running hyperparameter tuning, then you can specify a FLOAT64 value. The default value is 1.0.
            If you are running hyperparameter tuning, then you can use one of the following options:
                The HPARAM_RANGE keyword and two FLOAT64 values that define the range to use for the hyperparameter. For example, L2_REG = HPARAM_RANGE(1.5, 5.0).
                The HPARAM_CANDIDATES keyword and an array of FLOAT64 values that provide discrete values to use for the hyperparameter. For example, L2_REG = HPARAM_CANDIDATES([0, 1.0, 3.0, 5.0]).
    """

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series or pandas.core.frame.DataFrame or pandas.core.series.Series):
                Series or DataFrame of shape (n_samples, n_features). Training vector,
                where `n_samples` is the number of samples and `n_features` is
                the number of features.

            y (default None):
                Ignored.

        Returns:
            bigframes.ml.decomposition.MatrixFactorization: Fitted estimator.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def score(self, X=None, y=None):
        """Calculate evaluation metrics of the model.

        .. note::

            Output matches that of the BigQuery ML.EVALUATE function.
            See: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-evaluate#matrix_factorization_models
            for the outputs relevant to this model type.

        Args:
            X (default None):
                Ignored.

            y (default None):
                Ignored.
        Returns:
            bigframes.dataframe.DataFrame: DataFrame that represents model metrics.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def predict(self, X):
        """Predict the closest cluster for each sample in X.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series or pandas.core.frame.DataFrame or pandas.core.series.Series):
                Series or a DataFrame to predict.

        Returns:
            bigframes.dataframe.DataFrame: Predicted DataFrames."""
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
