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

from unittest import mock

import google.cloud.bigquery as bigquery
import pytest

import bigframes.ml.sql as ml_sql
import bigframes.pandas as bpd


@pytest.fixture(scope="session")
def base_sql_generator() -> ml_sql.BaseSqlGenerator:
    return ml_sql.BaseSqlGenerator()


@pytest.fixture(scope="session")
def model_creation_sql_generator() -> ml_sql.ModelCreationSqlGenerator:
    return ml_sql.ModelCreationSqlGenerator()


@pytest.fixture(scope="session")
def model_manipulation_sql_generator() -> ml_sql.ModelManipulationSqlGenerator:
    return ml_sql.ModelManipulationSqlGenerator(
        model_ref=bigquery.ModelReference.from_string(
            "my_project_id.my_dataset_id.my_model_id"
        )
    )


@pytest.fixture(scope="session")
def mock_df():
    mock_df = mock.create_autospec(spec=bpd.DataFrame)
    mock_df.sql = "input_X_y_sql"
    mock_df._to_sql_query.return_value = "input_X_sql", None, None

    return mock_df


def test_ml_arima_coefficients(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_arima_coefficients()
    assert (
        sql
        == """SELECT * FROM ML.ARIMA_COEFFICIENTS(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`)"""
    )


def test_options_correct(base_sql_generator: ml_sql.BaseSqlGenerator):
    sql = base_sql_generator.options(
        model_type="lin_reg", input_label_cols=["col_a"], l1_reg=0.6
    )
    assert (
        sql
        == """OPTIONS(
  model_type='lin_reg',
  input_label_cols=['col_a'],
  l1_reg=0.6)"""
    )


def test_transform_correct(base_sql_generator: ml_sql.BaseSqlGenerator):
    sql = base_sql_generator.transform(
        "ML.STANDARD_SCALER(col_a) OVER(col_a) AS scaled_col_a",
        "ML.ONE_HOT_ENCODER(col_b) OVER(col_b) AS encoded_col_b",
        "ML.LABEL_ENCODER(col_c) OVER(col_c) AS encoded_col_c",
    )
    assert (
        sql
        == """TRANSFORM(
  ML.STANDARD_SCALER(col_a) OVER(col_a) AS scaled_col_a,
  ML.ONE_HOT_ENCODER(col_b) OVER(col_b) AS encoded_col_b,
  ML.LABEL_ENCODER(col_c) OVER(col_c) AS encoded_col_c)"""
    )


def test_standard_scaler_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_standard_scaler("col_a", "scaled_col_a")
    assert sql == "ML.STANDARD_SCALER(`col_a`) OVER() AS `scaled_col_a`"


def test_max_abs_scaler_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_max_abs_scaler("col_a", "scaled_col_a")
    assert sql == "ML.MAX_ABS_SCALER(`col_a`) OVER() AS `scaled_col_a`"


def test_min_max_scaler_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_min_max_scaler("col_a", "scaled_col_a")
    assert sql == "ML.MIN_MAX_SCALER(`col_a`) OVER() AS `scaled_col_a`"


def test_imputer_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_imputer("col_a", "mean", "scaled_col_a")
    assert sql == "ML.IMPUTER(`col_a`, 'mean') OVER() AS `scaled_col_a`"


def test_k_bins_discretizer_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_bucketize("col_a", [1, 2, 3, 4], "scaled_col_a")
    assert sql == "ML.BUCKETIZE(`col_a`, [1, 2, 3, 4], FALSE) AS `scaled_col_a`"


def test_k_bins_discretizer_quantile_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_quantile_bucketize("col_a", 5, "scaled_col_a")
    assert sql == "ML.QUANTILE_BUCKETIZE(`col_a`, 5) OVER() AS `scaled_col_a`"


def test_one_hot_encoder_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_one_hot_encoder(
        "col_a", "none", 1000000, 0, "encoded_col_a"
    )
    assert (
        sql
        == "ML.ONE_HOT_ENCODER(`col_a`, 'none', 1000000, 0) OVER() AS `encoded_col_a`"
    )


def test_label_encoder_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_label_encoder("col_a", 1000000, 0, "encoded_col_a")
    assert sql == "ML.LABEL_ENCODER(`col_a`, 1000000, 0) OVER() AS `encoded_col_a`"


def test_polynomial_expand(
    base_sql_generator: ml_sql.BaseSqlGenerator,
):
    sql = base_sql_generator.ml_polynomial_expand(["col_a", "col_b"], 2, "poly_exp")
    assert sql == "ML.POLYNOMIAL_EXPAND(STRUCT(`col_a`, `col_b`), 2) AS `poly_exp`"


def test_ai_forecast_correct(
    base_sql_generator: ml_sql.BaseSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = base_sql_generator.ai_forecast(
        source_sql=mock_df.sql,
        options={
            "model": "TimesFM 2.0",
            "data_col": "data1",
            "timestamp_col": "time1",
            "id_cols": ("id1", "id2"),
            "horizon": 10,
            "confidence_level": 0.95,
        },
    )
    assert (
        sql
        == """SELECT * FROM AI.FORECAST((input_X_y_sql),
  model => 'TimesFM 2.0',
  data_col => 'data1',
  timestamp_col => 'time1',
  id_cols => ['id1', 'id2'],
  horizon => 10,
  confidence_level => 0.95)"""
    )


def test_create_model_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_creation_sql_generator.create_model(
        source_sql=mock_df.sql,
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_model_correct_sql"
        ),
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_model_correct_sql`
OPTIONS(
  option_key1='option_value1',
  option_key2=2)
AS input_X_y_sql"""
    )


def test_create_model_transform_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_creation_sql_generator.create_model(
        source_sql=mock_df.sql,
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_model_transform"
        ),
        options={"option_key1": "option_value1", "option_key2": 2},
        transforms=[
            "ML.STANDARD_SCALER(col_a) OVER(col_a) AS scaled_col_a",
            "ML.ONE_HOT_ENCODER(col_b) OVER(col_b) AS encoded_col_b",
        ],
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_model_transform`
TRANSFORM(
  ML.STANDARD_SCALER(col_a) OVER(col_a) AS scaled_col_a,
  ML.ONE_HOT_ENCODER(col_b) OVER(col_b) AS encoded_col_b)
OPTIONS(
  option_key1='option_value1',
  option_key2=2)
AS input_X_y_sql"""
    )


def test_create_llm_remote_model_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_creation_sql_generator.create_llm_remote_model(
        source_sql=mock_df.sql,
        connection_name="my_project.us.my_connection",
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_remote_model"
        ),
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_remote_model`
REMOTE WITH CONNECTION `my_project.us.my_connection`
OPTIONS(
  option_key1='option_value1',
  option_key2=2)
AS input_X_y_sql"""
    )


def test_create_remote_model_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
):
    sql = model_creation_sql_generator.create_remote_model(
        connection_name="my_project.us.my_connection",
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_remote_model"
        ),
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_remote_model`
REMOTE WITH CONNECTION `my_project.us.my_connection`
OPTIONS(
  option_key1='option_value1',
  option_key2=2)"""
    )


def test_create_remote_model_with_params_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
):
    sql = model_creation_sql_generator.create_remote_model(
        connection_name="my_project.us.my_connection",
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_remote_model"
        ),
        input={"column1": "int64"},
        output={"result": "array<float64>"},
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_remote_model`
INPUT(
  `column1` int64)
OUTPUT(
  `result` array<float64>)
REMOTE WITH CONNECTION `my_project.us.my_connection`
OPTIONS(
  option_key1='option_value1',
  option_key2=2)"""
    )


def test_create_imported_model_correct(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
):
    sql = model_creation_sql_generator.create_imported_model(
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_imported_model"
        ),
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_imported_model`
OPTIONS(
  option_key1='option_value1',
  option_key2=2)"""
    )


def test_create_xgboost_imported_model_produces_correct_sql(
    model_creation_sql_generator: ml_sql.ModelCreationSqlGenerator,
):
    sql = model_creation_sql_generator.create_xgboost_imported_model(
        model_ref=bigquery.ModelReference.from_string(
            "test-proj._anonXYZ.create_xgboost_imported_model"
        ),
        input={"column1": "int64"},
        output={"result": "array<float64>"},
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """CREATE OR REPLACE MODEL `test-proj`.`_anonXYZ`.`create_xgboost_imported_model`
INPUT(
  `column1` int64)
OUTPUT(
  `result` array<float64>)
OPTIONS(
  option_key1='option_value1',
  option_key2=2)"""
    )


def test_alter_model_correct_sql(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.alter_model(
        options={"option_key1": "option_value1", "option_key2": 2},
    )
    assert (
        sql
        == """ALTER MODEL `my_project_id`.`my_dataset_id`.`my_model_id`
SET OPTIONS(
  option_key1='option_value1',
  option_key2=2)"""
    )


def test_ml_predict_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_predict(source_sql=mock_df.sql)
    assert (
        sql
        == """SELECT * FROM ML.PREDICT(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  (input_X_y_sql))"""
    )


def test_ml_llm_evaluate_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_llm_evaluate(
        source_sql=mock_df.sql, task_type="CLASSIFICATION"
    )
    assert (
        sql
        == """SELECT * FROM ML.EVALUATE(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
            (input_X_y_sql), STRUCT("CLASSIFICATION" AS task_type))"""
    )


def test_ml_evaluate_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_evaluate(source_sql=mock_df.sql)
    assert (
        sql
        == """SELECT * FROM ML.EVALUATE(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  (input_X_y_sql))"""
    )


def test_ml_arima_evaluate_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_arima_evaluate(
        show_all_candidate_models=True
    )
    assert (
        sql
        == """SELECT * FROM ML.ARIMA_EVALUATE(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
            STRUCT(True AS show_all_candidate_models))"""
    )


def test_ml_evaluate_no_source_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_evaluate()
    assert (
        sql
        == """SELECT * FROM ML.EVALUATE(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`)"""
    )


def test_ml_centroids_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_centroids()
    assert (
        sql
        == """SELECT * FROM ML.CENTROIDS(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`)"""
    )


def test_ml_forecast_correct_sql(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_forecast(
        struct_options={"option_key1": 1, "option_key2": 2.2},
    )
    assert (
        sql
        == """SELECT * FROM ML.FORECAST(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  STRUCT(
  1 AS `option_key1`,
  2.2 AS `option_key2`))"""
    )


def test_ml_generate_text_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_generate_text(
        source_sql=mock_df.sql,
        struct_options={"option_key1": 1, "option_key2": 2.2},
    )
    assert (
        sql
        == """SELECT * FROM ML.GENERATE_TEXT(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  (input_X_y_sql), STRUCT(
  1 AS `option_key1`,
  2.2 AS `option_key2`))"""
    )


def test_ml_generate_embedding_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_generate_embedding(
        source_sql=mock_df.sql,
        struct_options={"option_key1": 1, "option_key2": 2.2},
    )
    assert (
        sql
        == """SELECT * FROM ML.GENERATE_EMBEDDING(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  (input_X_y_sql), STRUCT(
  1 AS `option_key1`,
  2.2 AS `option_key2`))"""
    )


def test_ml_explain_predict_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_explain_predict(
        source_sql=mock_df.sql,
        struct_options={"option_key1": 1, "option_key2": 2.25},
    )
    assert (
        sql
        == """SELECT * FROM ML.EXPLAIN_PREDICT(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  (input_X_y_sql), STRUCT(
  1 AS `option_key1`,
  2.25 AS `option_key2`))"""
    )


def test_ml_detect_anomalies_correct_sql(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
    mock_df: bpd.DataFrame,
):
    sql = model_manipulation_sql_generator.ml_detect_anomalies(
        source_sql=mock_df.sql,
        struct_options={"option_key1": 1, "option_key2": 2.2},
    )
    assert (
        sql
        == """SELECT * FROM ML.DETECT_ANOMALIES(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`,
  STRUCT(
  1 AS `option_key1`,
  2.2 AS `option_key2`), (input_X_y_sql))"""
    )


def test_ml_principal_components_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_principal_components()
    assert (
        sql
        == """SELECT * FROM ML.PRINCIPAL_COMPONENTS(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`)"""
    )


def test_ml_principal_component_info_correct(
    model_manipulation_sql_generator: ml_sql.ModelManipulationSqlGenerator,
):
    sql = model_manipulation_sql_generator.ml_principal_component_info()
    assert (
        sql
        == """SELECT * FROM ML.PRINCIPAL_COMPONENT_INFO(MODEL `my_project_id`.`my_dataset_id`.`my_model_id`)"""
    )
