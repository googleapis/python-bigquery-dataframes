"""Clustering models. This module is styled after Scikit-Learn's cluster module:
https://scikit-learn.org/stable/modules/clustering.html"""

from __future__ import annotations

from typing import cast, Optional, TYPE_CHECKING

from google.cloud import bigquery

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.core


class KMeans(bigframes.ml.api_primitives.BaseEstimator):
    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> KMeans:
        assert model.model_type == "KMEANS"

        kwargs = {}
        last_fitting = model.training_runs[-1]["trainingOptions"]
        if "numClusters" in last_fitting:
            kwargs["n_clusters"] = int(last_fitting["numClusters"])

        new_kmeans = KMeans(**kwargs)
        new_kmeans._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_kmeans

    def fit(self, X: bigframes.DataFrame):
        self._bqml_model = bigframes.ml.core.create_bqml_model(
            train_X=X,
            options={"model_type": "KMEANS", "num_clusters": self.n_clusters},
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        """Predict the closest cluster for each sample in X"""
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before predict")

        return cast(
            bigframes.dataframe.DataFrame, self._bqml_model.predict(X)[["CENTROID_ID"]]
        )

    def to_gbq(self, model_name: str, replace: bool = False) -> KMeans:
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before it can be saved")

        new_model = self._bqml_model.copy(model_name, replace)
        return new_model.session.read_gbq_model(self._bqml_model.model_name)
