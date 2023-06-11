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

import google.api_core.exceptions
import pandas
import pytest

import bigframes.ml.ensemble


def test_xgbregressor_model_eval(
    penguins_xgbregressor_model: bigframes.ml.ensemble.XGBRegressor,
):
    result = penguins_xgbregressor_model.score().compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [109.016973],
            "mean_squared_error": [20867.299758],
            "mean_squared_log_error": [0.00135],
            "median_absolute_error": [86.490234],
            "r2_score": [0.967458],
            "explained_variance": [0.967504],
        },
        dtype="Float64",
    )
    pandas.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-2,
        # int64 Index by default in pandas versus Int64 (nullable) Index in BigFramese
        check_index_type=False,
    )


def test_xgbregressor_model_score_with_data(
    penguins_xgbregressor_model, penguins_df_default_index
):
    df = penguins_df_default_index.dropna()
    test_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
    ]
    test_y = df[["sex"]]
    result = penguins_xgbregressor_model.score(test_X, test_y).compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [108.77582],
            "mean_squared_error": [20943.272738],
            "mean_squared_log_error": [0.00135],
            "median_absolute_error": [86.313477],
            "r2_score": [0.967571],
            "explained_variance": [0.967609],
        },
        dtype="Float64",
    )
    pandas.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-2,
        # int64 Index by default in pandas versus Int64 (nullable) Index in BigFramese
        check_index_type=False,
    )


def test_xgbregressor_model_predict(
    penguins_xgbregressor_model: bigframes.ml.ensemble.XGBRegressor, new_penguins_df
):
    result = penguins_xgbregressor_model.predict(new_penguins_df).compute()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": ["4293.1538089", "3410.0271", "3357.944"]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        result.sort_index(),
        expected,
        check_exact=False,
        rtol=1e-2,
        check_index_type=False,
    )


def test_to_gbq_saved_xgbregressor_model_scores(
    penguins_xgbregressor_model, dataset_id
):
    saved_model = penguins_xgbregressor_model.to_gbq(
        f"{dataset_id}.test_penguins_model", replace=True
    )
    result = saved_model.score().compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [109.016973],
            "mean_squared_error": [20867.299758],
            "mean_squared_log_error": [0.00135],
            "median_absolute_error": [86.490234],
            "r2_score": [0.967458],
            "explained_variance": [0.967504],
        },
        dtype="Float64",
    )
    pandas.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-2,
        # int64 Index by default in pandas versus Int64 (nullable) Index in BigFramese
        check_index_type=False,
    )


def test_to_xgbregressor_model_gbq_replace(penguins_xgbregressor_model, dataset_id):
    penguins_xgbregressor_model.to_gbq(
        f"{dataset_id}.test_penguins_model", replace=True
    )
    with pytest.raises(google.api_core.exceptions.Conflict):
        penguins_xgbregressor_model.to_gbq(f"{dataset_id}.test_penguins_model")


def test_xgbclassifier_model_eval(
    penguins_xgbclassifier_model: bigframes.ml.ensemble.XGBClassifier,
):
    result = penguins_xgbclassifier_model.score().compute()
    expected = pandas.DataFrame(
        {
            "precision": [1.0],
            "recall": [1.0],
            "accuracy": [1.0],
            "f1_score": [1.0],
            "log_loss": [0.331442],
            "roc_auc": [1.0],
        },
        dtype="Float64",
    )
    pandas.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-2,
        # int64 Index by default in pandas versus Int64 (nullable) Index in BigFramese
        check_index_type=False,
    )


def test_xgbclassifier_model_score_with_data(
    penguins_xgbclassifier_model, penguins_df_default_index
):
    df = penguins_df_default_index.dropna()
    test_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
    ]
    test_y = df[["sex"]]
    result = penguins_xgbclassifier_model.score(test_X, test_y).compute()
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


def test_xgbclassifier_model_predict(
    penguins_xgbclassifier_model: bigframes.ml.ensemble.XGBClassifier, new_penguins_df
):
    result = penguins_xgbclassifier_model.predict(new_penguins_df).compute()
    expected = pandas.DataFrame(
        {"predicted_sex": ["MALE", "MALE", "FEMALE"]},
        dtype="string[pyarrow]",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        result.sort_index(),
        expected,
        check_exact=False,
        rtol=1e-2,
        check_index_type=False,
    )


def test_to_gbq_saved_xgclassifier_model_scores(
    penguins_xgbclassifier_model, dataset_id
):
    saved_model = penguins_xgbclassifier_model.to_gbq(
        f"{dataset_id}.test_penguins_model", replace=True
    )
    result = saved_model.score().compute()
    expected = pandas.DataFrame(
        {
            "precision": [1.0],
            "recall": [1.0],
            "accuracy": [1.0],
            "f1_score": [1.0],
            "log_loss": [0.331442],
            "roc_auc": [1.0],
        },
        dtype="Float64",
    )
    pandas.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        rtol=1e-2,
        # int64 Index by default in pandas versus Int64 (nullable) Index in BigFramese
        check_index_type=False,
    )


def test_to_xgclassifier_model_gbq_replace(penguins_xgbclassifier_model, dataset_id):
    penguins_xgbclassifier_model.to_gbq(
        f"{dataset_id}.test_penguins_model", replace=True
    )
    with pytest.raises(google.api_core.exceptions.Conflict):
        penguins_xgbclassifier_model.to_gbq(f"{dataset_id}.test_penguins_model")
