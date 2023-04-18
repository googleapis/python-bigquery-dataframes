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

import hashlib
import logging
from typing import cast

import google.cloud.exceptions
import pytest

import bigframes.ml.cluster
import bigframes.ml.core
import bigframes.ml.linear_model


@pytest.fixture(scope="session")
def penguins_bqml_linear_model(
    session, penguins_linear_model_name
) -> bigframes.ml.core.BqmlModel:
    model = session.bqclient.get_model(penguins_linear_model_name)
    return bigframes.ml.core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def penguins_linear_model(
    session, penguins_linear_model_name
) -> bigframes.ml.linear_model.LinearRegression:
    return cast(
        bigframes.ml.linear_model.LinearRegression,
        session.read_gbq_model(penguins_linear_model_name),
    )


@pytest.fixture(scope="session")
def penguins_kmeans_model(
    session: bigframes.Session, dataset_id_permanent, penguins_table_id
) -> bigframes.ml.cluster.KMeans:
    """Provides a pretrained model as a test fixture that is cached across test runs.
    This lets us run system tests without having to wait for a model.fit(...)"""
    sql = f"""
CREATE OR REPLACE MODEL `$model_name`
OPTIONS (
    model_type='kmeans',
    num_clusters=3
) AS SELECT
    culmen_length_mm,
    culmen_depth_mm,
    flipper_length_mm,
    sex
FROM `{penguins_table_id}`"""
    # We use the SQL hash as the name to ensure the model is regenerated if this fixture is edited
    model_name = f"{dataset_id_permanent}.penguins_cluster_{hashlib.md5(sql.encode()).hexdigest()}"
    sql = sql.replace("$model_name", model_name)

    try:
        return session.read_gbq_model(model_name)
    except google.cloud.exceptions.NotFound:
        logging.info(
            "penguins_linear_model fixture was not found in the permanent dataset, regenerating it..."
        )
        session.bqclient.query(sql).result()
        return session.read_gbq_model(model_name)
