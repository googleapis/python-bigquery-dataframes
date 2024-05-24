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

import bigframes.ml.linear_model
from tests.system import utils


def test_linear_regression_configure_fit_score(penguins_df_default_index, dataset_id):
    model = bigframes.ml.linear_model.LinearRegression()

    df = penguins_df_default_index.dropna()
    X_train = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    y_train = df[["body_mass_g"]]
    model.fit(X_train, y_train)

    # Check score to ensure the model was fitted
    result = model.score(X_train, y_train).to_pandas()
    utils.check_pandas_df_schema_and_index(
        result, columns=utils.ML_REGRESSION_METRICS, index=1
    )

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(f"{dataset_id}.temp_configured_model", replace=True)
    assert reloaded_model._bqml_model is not None
    assert (
        f"{dataset_id}.temp_configured_model" in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.optimize_strategy == "NORMAL_EQUATION"
    assert reloaded_model.fit_intercept is True
    assert reloaded_model.calculate_p_values is False
    assert reloaded_model.enable_global_explain is False
    assert reloaded_model.l1_reg is None
    assert reloaded_model.l2_reg == 0.0
    assert reloaded_model.learning_rate is None
    assert reloaded_model.learning_rate_strategy == "line_search"
    assert reloaded_model.ls_init_learning_rate is None
    assert reloaded_model.max_iterations == 20
    assert reloaded_model.tol == 0.01


def test_linear_regression_customized_params_fit_score(
    penguins_df_default_index, dataset_id
):
    model = bigframes.ml.linear_model.LinearRegression(
        fit_intercept=False,
        l2_reg=0.2,
        tol=0.02,
        l1_reg=0.2,
        max_iterations=30,
        optimize_strategy="batch_gradient_descent",
        learning_rate_strategy="constant",
        learning_rate=0.2,
    )

    df = penguins_df_default_index.dropna()
    X_train = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    y_train = df[["body_mass_g"]]
    model.fit(X_train, y_train)

    # Check score to ensure the model was fitted
    result = model.score(X_train, y_train).to_pandas()
    utils.check_pandas_df_schema_and_index(
        result, columns=utils.ML_REGRESSION_METRICS, index=1
    )

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(f"{dataset_id}.temp_configured_model", replace=True)
    assert reloaded_model._bqml_model is not None
    assert (
        f"{dataset_id}.temp_configured_model" in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.optimize_strategy == "BATCH_GRADIENT_DESCENT"
    assert reloaded_model.fit_intercept is False
    assert reloaded_model.calculate_p_values is False
    assert reloaded_model.enable_global_explain is False
    assert reloaded_model.l1_reg == 0.2
    assert reloaded_model.l2_reg == 0.2
    assert reloaded_model.ls_init_learning_rate is None
    assert reloaded_model.max_iterations == 30
    assert reloaded_model.tol == 0.02
    assert reloaded_model.learning_rate_strategy == "CONSTANT"
    assert reloaded_model.learning_rate == 0.2


# TODO(garrettwu): add tests for param warm_start. Requires a trained model.


def test_logistic_regression_configure_fit_score(penguins_df_default_index, dataset_id):
    model = bigframes.ml.linear_model.LogisticRegression()

    df = penguins_df_default_index.dropna()
    X_train = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
    ]
    y_train = df[["sex"]]
    model.fit(X_train, y_train)

    # Check score to ensure the model was fitted
    result = model.score(X_train, y_train).to_pandas()
    utils.check_pandas_df_schema_and_index(
        result, columns=utils.ML_CLASSFICATION_METRICS, index=1
    )

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_logistic_reg_model", replace=True
    )
    assert reloaded_model._bqml_model is not None
    assert (
        f"{dataset_id}.temp_configured_logistic_reg_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.fit_intercept is True
    assert reloaded_model.class_weight is None


def test_logistic_regression_customized_params_fit_score(
    penguins_df_default_index, dataset_id
):
    model = bigframes.ml.linear_model.LogisticRegression(
        fit_intercept=False,
        class_weight="balanced",
        l2_reg=0.2,
        tol=0.02,
        l1_reg=0.2,
        max_iterations=30,
        optimize_strategy="batch_gradient_descent",
        learning_rate_strategy="constant",
        learning_rate=0.2,
    )
    df = penguins_df_default_index.dropna()
    X_train = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    y_train = df[["sex"]]
    model.fit(X_train, y_train)

    # Check score to ensure the model was fitted
    result = model.score(X_train, y_train).to_pandas()
    utils.check_pandas_df_schema_and_index(
        result, columns=utils.ML_CLASSFICATION_METRICS, index=1
    )

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_logistic_reg_model", replace=True
    )
    assert reloaded_model._bqml_model is not None
    assert (
        f"{dataset_id}.temp_configured_logistic_reg_model"
        in reloaded_model._bqml_model.model_name
    )
    # TODO(garrettwu) optimize_strategy isn't logged in BQML
    # assert reloaded_model.optimize_strategy == "BATCH_GRADIENT_DESCENT"
    assert reloaded_model.fit_intercept is False
    assert reloaded_model.class_weight == "balanced"
    assert reloaded_model.calculate_p_values is False
    assert reloaded_model.enable_global_explain is False
    assert reloaded_model.l1_reg == 0.2
    assert reloaded_model.l2_reg == 0.2
    assert reloaded_model.ls_init_learning_rate is None
    assert reloaded_model.max_iterations == 30
    assert reloaded_model.tol == 0.02
    assert reloaded_model.learning_rate_strategy == "CONSTANT"
    assert reloaded_model.learning_rate == 0.2
