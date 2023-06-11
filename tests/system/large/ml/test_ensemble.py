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

from unittest import TestCase

import pandas

import bigframes.ml.ensemble


def test_xgbregressor_auto_split(penguins_df_default_index, dataset_id):
    # Note: with <500 data points, AUTO_SPLIT will behave equivalently to NO_SPLIT
    model = bigframes.ml.ensemble.XGBRegressor(
        data_split_method="AUTO_SPLIT",
    )

    df = penguins_df_default_index.dropna()
    train_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    train_y = df[["body_mass_g"]]
    model.fit(train_X, train_y)

    # Check score to ensure the model was fitted
    result = model.score().compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [97.368139],
            "mean_squared_error": [16284.877027],
            "mean_squared_log_error": [0.0010189],
            "median_absolute_error": [72.158691],
            "r2_score": [0.974784],
            "explained_variance": [0.974845],
        },
        dtype="Float64",
    )
    expected = expected.reindex(index=expected.index.astype("Int64"))
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_xgbregressor_model", replace=True
    )
    assert (
        f"{dataset_id}.temp_configured_xgbregressor_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.data_split_method == "AUTO_SPLIT"


def test_xgbregressor_dart_booster_num_parallel_tree(
    penguins_df_default_index, dataset_id
):
    # Note: with <500 data points, AUTO_SPLIT will behave equivalently to NO_SPLIT
    model = bigframes.ml.ensemble.XGBRegressor(
        booster_type="dart",
        num_parallel_tree=2,
        early_stop=True,
        subsample=0.2,
    )

    df = penguins_df_default_index.dropna().sample(n=70)
    train_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    train_y = df[["body_mass_g"]]
    model.fit(train_X, train_y)

    # Check score to ensure the model was fitted
    result = model.score().compute()
    TestCase().assertSequenceEqual(result.shape, (1, 6))
    assert 10000 <= int(result.mean_squared_error) <= 120000
    for col_name in [
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
        "r2_score",
        "explained_variance",
    ]:
        assert col_name in result.columns

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_xgbregressor_model", replace=True
    )
    assert (
        f"{dataset_id}.temp_configured_xgbregressor_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.booster_type == "DART"
    assert reloaded_model.num_parallel_tree == 2
    assert reloaded_model.early_stop is True
    assert reloaded_model.subsample == 0.2


def test_xgbclassifier_auto_split(penguins_df_default_index, dataset_id):
    # Note: with <500 data points, AUTO_SPLIT will behave equivalently to NO_SPLIT
    model = bigframes.ml.ensemble.XGBClassifier(
        data_split_method="AUTO_SPLIT",
    )

    df = penguins_df_default_index.dropna().sample(n=70)
    train_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    train_y = df[["sex"]]
    model.fit(train_X, train_y)

    # Check score to ensure the model was fitted
    result = model.score().compute()
    # Check score to ensure the model was fitted
    result = model.score().compute()
    TestCase().assertSequenceEqual(result.shape, (1, 6))
    for col_name in [
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "log_loss",
        "roc_auc",
    ]:
        assert col_name in result.columns

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_xgbclassifierr_model", replace=True
    )
    assert (
        f"{dataset_id}.temp_configured_xgbclassifierr_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.data_split_method == "AUTO_SPLIT"


def test_xgbclassifier_dart_booster_num_parallel_tree(
    penguins_df_default_index, dataset_id
):
    # Note: with <500 data points, AUTO_SPLIT will behave equivalently to NO_SPLIT
    model = bigframes.ml.ensemble.XGBClassifier(
        booster_type="dart",
        num_parallel_tree=2,
        early_stop=True,
        subsample=0.2,
    )

    df = penguins_df_default_index.dropna().sample(n=70)
    train_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    train_y = df[["sex"]]
    model.fit(train_X, train_y)

    # Check score to ensure the model was fitted
    result = model.score().compute()
    # Check score to ensure the model was fitted
    result = model.score().compute()
    TestCase().assertSequenceEqual(result.shape, (1, 6))
    for col_name in [
        "precision",
        "recall",
        "accuracy",
        "f1_score",
        "log_loss",
        "roc_auc",
    ]:
        assert col_name in result.columns

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(
        f"{dataset_id}.temp_configured_xgbclassifierr_model", replace=True
    )
    assert (
        f"{dataset_id}.temp_configured_xgbclassifierr_model"
        in reloaded_model._bqml_model.model_name
    )
    assert reloaded_model.booster_type == "DART"
    assert reloaded_model.num_parallel_tree == 2
    assert reloaded_model.early_stop is True
    assert reloaded_model.subsample == 0.2
