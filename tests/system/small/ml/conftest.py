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

import os
from typing import cast
import uuid

import pandas as pd
import pytest

import bigframes
from bigframes.ml import (
    cluster,
    core,
    decomposition,
    ensemble,
    forecasting,
    globals,
    imported,
    linear_model,
    remote,
)


@pytest.fixture(scope="session")
def penguins_bqml_linear_model(session, penguins_linear_model_name) -> core.BqmlModel:
    model = session.bqclient.get_model(penguins_linear_model_name)
    return core.BqmlModel(session, model)


@pytest.fixture(scope="function")
def ephemera_penguins_bqml_linear_model(
    session: bigframes.Session,
    penguins_bqml_linear_model: core.BqmlModel,
) -> core.BqmlModel:
    model = penguins_bqml_linear_model
    return model.copy(f"{session._anonymous_dataset}.{uuid.uuid4().hex}")


@pytest.fixture(scope="session")
def penguins_bqml_kmeans_model(
    session: bigframes.Session, penguins_kmeans_model_name: str
) -> core.BqmlModel:
    model = session.bqclient.get_model(penguins_kmeans_model_name)
    return core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def penguins_bqml_pca_model(
    session: bigframes.Session, penguins_pca_model_name: str
) -> core.BqmlModel:
    model = session.bqclient.get_model(penguins_pca_model_name)
    return core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def penguins_linear_model(
    session, penguins_linear_model_name: str
) -> linear_model.LinearRegression:
    return cast(
        linear_model.LinearRegression,
        session.read_gbq_model(penguins_linear_model_name),
    )


@pytest.fixture(scope="function")
def ephemera_penguins_linear_model(
    ephemera_penguins_bqml_linear_model: core.BqmlModel,
) -> linear_model.LinearRegression:
    bf_model = linear_model.LinearRegression()
    bf_model._bqml_model = ephemera_penguins_bqml_linear_model
    return bf_model


@pytest.fixture(scope="function")
def penguins_linear_model_w_global_explain(
    penguins_bqml_linear_model: core.BqmlModel,
) -> linear_model.LinearRegression:
    bf_model = linear_model.LinearRegression(enable_global_explain=True)
    bf_model._bqml_model = penguins_bqml_linear_model
    return bf_model


@pytest.fixture(scope="session")
def penguins_logistic_model(
    session, penguins_logistic_model_name
) -> linear_model.LogisticRegression:
    return cast(
        linear_model.LogisticRegression,
        session.read_gbq_model(penguins_logistic_model_name),
    )


@pytest.fixture(scope="session")
def penguins_xgbregressor_model(
    session, penguins_xgbregressor_model_name
) -> ensemble.XGBRegressor:
    return cast(
        ensemble.XGBRegressor,
        session.read_gbq_model(penguins_xgbregressor_model_name),
    )


@pytest.fixture(scope="session")
def penguins_xgbclassifier_model(
    session, penguins_xgbclassifier_model_name
) -> ensemble.XGBClassifier:
    return cast(
        ensemble.XGBClassifier,
        session.read_gbq_model(penguins_xgbclassifier_model_name),
    )


@pytest.fixture(scope="session")
def penguins_randomforest_regressor_model(
    session, penguins_randomforest_regressor_model_name
) -> ensemble.RandomForestRegressor:
    return cast(
        ensemble.RandomForestRegressor,
        session.read_gbq_model(penguins_randomforest_regressor_model_name),
    )


@pytest.fixture(scope="session")
def penguins_randomforest_classifier_model(
    session, penguins_randomforest_classifier_model_name
) -> ensemble.RandomForestClassifier:
    return cast(
        ensemble.RandomForestClassifier,
        session.read_gbq_model(penguins_randomforest_classifier_model_name),
    )


@pytest.fixture(scope="session")
def penguins_kmeans_model(session, penguins_kmeans_model_name: str) -> cluster.KMeans:
    return cast(
        cluster.KMeans,
        session.read_gbq_model(penguins_kmeans_model_name),
    )


@pytest.fixture(scope="session")
def penguins_pca_model(
    session: bigframes.Session, penguins_pca_model_name: str
) -> decomposition.PCA:
    return cast(
        decomposition.PCA,
        session.read_gbq_model(penguins_pca_model_name),
    )


@pytest.fixture(scope="session")
def onnx_iris_pandas_df():
    """Data matching the iris dataset."""
    return pd.DataFrame(
        {
            "sepal_length": [4.9, 5.1, 34.7],
            "sepal_width": [3.0, 5.1, 24.7],
            "petal_length": [1.4, 1.5, 13.3],
            "petal_width": [0.4, 0.2, 18.3],
            "species": [
                "setosa",
                "setosa",
                "virginica",
            ],
        }
    )


@pytest.fixture(scope="session")
def xgboost_iris_pandas_df():
    """Data matching the iris dataset."""
    return pd.DataFrame(
        {
            "sepal_length": [4.9, 5.1, 34.7],
            "sepal_width": [3.0, 5.1, 24.7],
            "petal_length": [1.4, 1.5, 13.3],
            "petal_width": [0.4, 0.2, 18.3],
        }
    )


@pytest.fixture(scope="session")
def onnx_iris_df(session, onnx_iris_pandas_df):
    return session.read_pandas(onnx_iris_pandas_df)


@pytest.fixture(scope="session")
def xgboost_iris_df(session, xgboost_iris_pandas_df):
    return session.read_pandas(xgboost_iris_pandas_df)


@pytest.fixture(scope="session")
def linear_remote_model_params() -> dict:
    # Pre-deployed endpoint of linear reg model in Vertex.
    # bigframes-test-linreg2 -> bigframes-test-linreg-endpoint2
    model_vertex_endpoint = os.environ.get(
        "BIGFRAMES_TEST_MODEL_VERTEX_ENDPOINT",
        "https://us-central1-aiplatform.googleapis.com/v1/projects/1084210331973/locations/us-central1/endpoints/3193318217619603456",
    )

    return {
        "input": {"culmen_length_mm": "float64"},
        "output": {"predicted_body_mass_g": "array<float64>"},
        "endpoint": model_vertex_endpoint,
    }


@pytest.fixture(scope="session")
def bqml_linear_remote_model(
    session, bq_connection, linear_remote_model_params
) -> core.BqmlModel:
    options = {
        "endpoint": linear_remote_model_params["endpoint"],
    }
    return globals.bqml_model_factory().create_remote_model(
        session=session,
        input=linear_remote_model_params["input"],
        output=linear_remote_model_params["output"],
        connection_name=bq_connection,
        options=options,
    )


@pytest.fixture(scope="session")
def linear_remote_vertex_model(
    session, bq_connection, linear_remote_model_params
) -> remote.VertexAIModel:
    return remote.VertexAIModel(
        endpoint=linear_remote_model_params["endpoint"],
        input=linear_remote_model_params["input"],
        output=linear_remote_model_params["output"],
        session=session,
        connection_name=bq_connection,
    )


@pytest.fixture(scope="session")
def time_series_bqml_arima_plus_model(
    session, time_series_arima_plus_model_name
) -> core.BqmlModel:
    model = session.bqclient.get_model(time_series_arima_plus_model_name)
    return core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def time_series_bqml_arima_plus_model_w_id(
    session, time_series_arima_plus_model_name_w_id
) -> core.BqmlModel:
    model = session.bqclient.get_model(time_series_arima_plus_model_name_w_id)
    return core.BqmlModel(session, model)


@pytest.fixture(scope="session")
def time_series_arima_plus_model(
    session, time_series_arima_plus_model_name
) -> forecasting.ARIMAPlus:
    return cast(
        forecasting.ARIMAPlus,
        session.read_gbq_model(time_series_arima_plus_model_name),
    )


@pytest.fixture(scope="session")
def time_series_arima_plus_model_w_id(
    session, time_series_arima_plus_model_name_w_id
) -> forecasting.ARIMAPlus:
    return cast(
        forecasting.ARIMAPlus,
        session.read_gbq_model(time_series_arima_plus_model_name_w_id),
    )


@pytest.fixture(scope="session")
def imported_tensorflow_model_path() -> str:
    return "gs://cloud-training-demos/txtclass/export/exporter/1549825580/*"


@pytest.fixture(scope="session")
def imported_onnx_model_path() -> str:
    return "gs://cloud-samples-data/bigquery/ml/onnx/pipeline_rf.onnx"


@pytest.fixture(scope="session")
def imported_xgboost_array_model_path() -> str:
    return "gs://bigframes-dev-testing/xgboost-testdata/model.bst"


@pytest.fixture(scope="session")
def imported_tensorflow_model(
    session, imported_tensorflow_model_path
) -> imported.TensorFlowModel:
    return imported.TensorFlowModel(
        session=session,
        model_path=imported_tensorflow_model_path,
    )


@pytest.fixture(scope="function")
def ephemera_imported_tensorflow_model(session) -> imported.TensorFlowModel:
    return imported.TensorFlowModel(
        session=session,
        model_path="gs://cloud-training-demos/txtclass/export/exporter/1549825580/*",
    )


@pytest.fixture(scope="session")
def imported_onnx_model(session, imported_onnx_model_path) -> imported.ONNXModel:
    return imported.ONNXModel(
        session=session,
        model_path=imported_onnx_model_path,
    )


@pytest.fixture(scope="session")
def imported_xgboost_model(
    session, imported_xgboost_array_model_path
) -> imported.XGBoostModel:
    return imported.XGBoostModel(
        session=session,
        input={
            "petal_length": "float64",
            "petal_width": "float64",
            "sepal_length": "float64",
            "sepal_width": "float64",
        },
        output={"predicted_label": "float64"},
        model_path=imported_xgboost_array_model_path,
    )
