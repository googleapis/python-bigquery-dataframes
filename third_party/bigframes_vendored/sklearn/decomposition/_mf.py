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

# from bigframes import constants


class PCA(BaseEstimator, metaclass=ABCMeta):
    """Matrix Factorization (MF).

    **Examples:**

        >>> import numpy as np
        >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
        >>> from sklearn.decomposition import NMF
        >>> model = NMF(n_components=2, init='random', random_state=0)
        >>> W = model.fit_transform(X)
        >>> H = model.components_

    Args:
        n_components (int, float or None, default None):
            Number of components to keep. If n_components is not set, all
            components are kept, n_components = min(n_samples, n_features).
            If 0 < n_components < 1, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
        svd_solver ("full", "randomized" or "auto", default "auto"):
            The solver to use to calculate the principal components. Details: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create-pca#pca_solver.

    """
