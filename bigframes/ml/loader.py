# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Union

from google.cloud import bigquery

import bigframes
import bigframes.constants as constants
from bigframes.ml import cluster, decomposition, ensemble, forecasting, linear_model


def from_bq(
    session: bigframes.Session, model: bigquery.Model
) -> Union[
    decomposition.PCA,
    cluster.KMeans,
    linear_model.LinearRegression,
    linear_model.LogisticRegression,
    ensemble.XGBRegressor,
    ensemble.XGBClassifier,
    forecasting.ARIMAPlus,
    ensemble.RandomForestRegressor,
    ensemble.RandomForestClassifier,
]:
    """Load a BQML model to BigQuery DataFrames ML.

    Args:
        session: a BigQuery DataFrames session.
        model: a BigQuery model.

    Returns:
        A BigQuery DataFrames ML model object.
    """
    if model.model_type == "LINEAR_REGRESSION":
        return linear_model.LinearRegression._from_bq(session, model)
    elif model.model_type == "KMEANS":
        return cluster.KMeans._from_bq(session, model)
    elif model.model_type == "PCA":
        return decomposition.PCA._from_bq(session, model)
    elif model.model_type == "LOGISTIC_REGRESSION":
        return linear_model.LogisticRegression._from_bq(session, model)
    elif model.model_type == "BOOSTED_TREE_REGRESSOR":
        return ensemble.XGBRegressor._from_bq(session, model)
    elif model.model_type == "BOOSTED_TREE_CLASSIFIER":
        return ensemble.XGBClassifier._from_bq(session, model)
    elif model.model_type == "ARIMA_PLUS":
        return forecasting.ARIMAPlus._from_bq(session, model)
    elif model.model_type == "RANDOM_FOREST_REGRESSOR":
        return ensemble.RandomForestRegressor._from_bq(session, model)
    elif model.model_type == "RANDOM_FOREST_CLASSIFIER":
        return ensemble.RandomForestClassifier._from_bq(session, model)
    else:
        raise NotImplementedError(
            f"Model type {model.model_type} is not yet supported by BigQuery DataFrames. {constants.FEEDBACK_LINK}"
        )
