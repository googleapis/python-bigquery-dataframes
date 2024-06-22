"""
This file contains preprocessing tools based on polynomials.
"""

from bigframes_vendored.sklearn.base import BaseEstimator, TransformerMixin

from bigframes import constants


class PolynomialFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial and interaction features."""

    def fit(self, X, y=None):
        """Compute number of output features.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series):
                The Dataframe or Series with training data.

            y (default None):
                Ignored.

        Returns:
            PolynomialFeatures: Fitted transformer.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    def transform(self, X):
        """Transform data to polynomial features.

        Args:
            X (bigframes.dataframe.DataFrame or bigframes.series.Series):
                The DataFrame or Series to be transformed.

        Returns:
           bigframes.dataframe.DataFrame: Transformed result.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
