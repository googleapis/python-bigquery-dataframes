# Copyright 2024 Google LLC
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

from unittest import mock

import pandas as pd
import pytest

import bigframes.bigquery._operations.ml as ml_ops
import bigframes.session


@pytest.fixture
def mock_session():
    return mock.create_autospec(spec=bigframes.session.Session)


MODEL_SERIES = pd.Series(
    {
        "modelReference": {
            "projectId": "test-project",
            "datasetId": "test-dataset",
            "modelId": "test-model",
        }
    }
)

MODEL_NAME = "test-project.test-dataset.test-model"


def test_get_model_name_and_session_with_pandas_series_model_input():
    model_name, _ = ml_ops._get_model_name_and_session(MODEL_SERIES)
    assert model_name == MODEL_NAME


def test_get_model_name_and_session_with_pandas_series_model_input_missing_model_reference():
    model_series = pd.Series({"some_other_key": "value"})
    with pytest.raises(
        ValueError, match="modelReference must be present in the pandas Series"
    ):
        ml_ops._get_model_name_and_session(model_series)


@mock.patch("bigframes.pandas.read_pandas")
def test_to_sql_with_pandas_dataframe(read_pandas_mock):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    read_pandas_mock.return_value._to_sql_query.return_value = (
        "SELECT * FROM `pandas_df`",
        [],
        [],
    )
    ml_ops._to_sql(df)
    read_pandas_mock.assert_called_once()


@mock.patch("bigframes.pandas.read_pandas")
@mock.patch("bigframes.core.sql.ml.create_model_ddl")
def test_create_model_with_pandas_dataframe(
    create_model_ddl_mock, read_pandas_mock, mock_session
):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    read_pandas_mock.return_value._to_sql_query.return_value = (
        "SELECT * FROM `pandas_df`",
        [],
        [],
    )
    ml_ops.create_model("model_name", training_data=df, session=mock_session)
    read_pandas_mock.assert_called_once()
    create_model_ddl_mock.assert_called_once()


@mock.patch("bigframes.pandas.read_gbq_query")
@mock.patch("bigframes.pandas.read_pandas")
@mock.patch("bigframes.core.sql.ml.evaluate")
def test_evaluate_with_pandas_dataframe(
    evaluate_mock, read_pandas_mock, read_gbq_query_mock
):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    read_pandas_mock.return_value._to_sql_query.return_value = (
        "SELECT * FROM `pandas_df`",
        [],
        [],
    )
    evaluate_mock.return_value = "SELECT * FROM `pandas_df`"
    ml_ops.evaluate(MODEL_SERIES, input_=df)
    read_pandas_mock.assert_called_once()
    evaluate_mock.assert_called_once()
    read_gbq_query_mock.assert_called_once_with("SELECT * FROM `pandas_df`")


@mock.patch("bigframes.pandas.read_gbq_query")
@mock.patch("bigframes.pandas.read_pandas")
@mock.patch("bigframes.core.sql.ml.predict")
def test_predict_with_pandas_dataframe(
    predict_mock, read_pandas_mock, read_gbq_query_mock
):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    read_pandas_mock.return_value._to_sql_query.return_value = (
        "SELECT * FROM `pandas_df`",
        [],
        [],
    )
    predict_mock.return_value = "SELECT * FROM `pandas_df`"
    ml_ops.predict(MODEL_SERIES, input_=df)
    read_pandas_mock.assert_called_once()
    predict_mock.assert_called_once()
    read_gbq_query_mock.assert_called_once_with("SELECT * FROM `pandas_df`")


@mock.patch("bigframes.pandas.read_gbq_query")
@mock.patch("bigframes.pandas.read_pandas")
@mock.patch("bigframes.core.sql.ml.explain_predict")
def test_explain_predict_with_pandas_dataframe(
    explain_predict_mock, read_pandas_mock, read_gbq_query_mock
):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    read_pandas_mock.return_value._to_sql_query.return_value = (
        "SELECT * FROM `pandas_df`",
        [],
        [],
    )
    explain_predict_mock.return_value = "SELECT * FROM `pandas_df`"
    ml_ops.explain_predict(MODEL_SERIES, input_=df)
    read_pandas_mock.assert_called_once()
    explain_predict_mock.assert_called_once()
    read_gbq_query_mock.assert_called_once_with("SELECT * FROM `pandas_df`")


@mock.patch("bigframes.pandas.read_gbq_query")
@mock.patch("bigframes.core.sql.ml.global_explain")
def test_global_explain_with_pandas_series_model(
    global_explain_mock, read_gbq_query_mock
):
    global_explain_mock.return_value = "SELECT * FROM `pandas_df`"
    ml_ops.global_explain(MODEL_SERIES)
    global_explain_mock.assert_called_once()
    read_gbq_query_mock.assert_called_once_with("SELECT * FROM `pandas_df`")
