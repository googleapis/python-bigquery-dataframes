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

from typing import TYPE_CHECKING, Union

from google.cloud import bigquery

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.cluster
import bigframes.ml.decomposition
import bigframes.ml.ensemble
import bigframes.ml.linear_model


def from_bq(
    session: bigframes.Session, model: bigquery.Model
) -> Union[
    bigframes.ml.decomposition.PCA,
    bigframes.ml.cluster.KMeans,
    bigframes.ml.linear_model.LinearRegression,
    bigframes.ml.linear_model.LogisticRegression,
    bigframes.ml.ensemble.XGBRegressor,
]:
    """Load a BQML model to BigFrames ML.

    Args:
        session: a BigFrames session.
        model: a BigQuery model.

    Returns: a BigFrames ML model object."""
    if model.model_type == "LINEAR_REGRESSION":
        return bigframes.ml.linear_model.LinearRegression._from_bq(session, model)
    elif model.model_type == "KMEANS":
        return bigframes.ml.cluster.KMeans._from_bq(session, model)
    elif model.model_type == "PCA":
        return bigframes.ml.decomposition.PCA._from_bq(session, model)
    elif model.model_type == "LOGISTIC_REGRESSION":
        return bigframes.ml.linear_model.LogisticRegression._from_bq(session, model)
    elif model.model_type == "BOOSTED_TREE_REGRESSOR":
        return bigframes.ml.ensemble.XGBRegressor._from_bq(session, model)
    else:
        raise NotImplementedError(
            f"Model type {model.model_type} is not yet supported by BigFrames"
        )
