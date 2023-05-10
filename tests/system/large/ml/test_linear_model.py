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

import pandas

import bigframes.ml.linear_model


def test_linear_regression_auto_split_configure_fit_score(
    penguins_df_default_index, dataset_id
):
    # Note: with <500 data points, AUTO_SPLIT will behave equivalently to NO_SPLIT
    model = bigframes.ml.linear_model.LinearRegression(
        data_split_method="AUTO_SPLIT", fit_intercept=False
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
            "mean_absolute_error": [225.735767],
            "mean_squared_error": [80417.461828],
            "mean_squared_log_error": [0.004967],
            "median_absolute_error": [172.543702],
            "r2_score": [0.87548],
            "explained_variance": [0.87548],
        },
        dtype="Float64",
    )
    expected = expected.reindex(index=expected.index.astype("Int64"))
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(f"{dataset_id}.temp_configured_model", replace=True)
    assert reloaded_model.data_split_method == "AUTO_SPLIT"

    # TODO(yunmengxie): enable this once b/277242951 (fit_intercept missing from API) is fixed
    # assert reloaded_model.fit_intercept == False


def test_linear_regression_manual_split_configure_fit_score(
    penguins_df_default_index, dataset_id
):
    model = bigframes.ml.linear_model.LinearRegression(
        data_split_method="NO_SPLIT", fit_intercept=True
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
            "mean_absolute_error": [225.735767],
            "mean_squared_error": [80417.461828],
            "mean_squared_log_error": [0.004967],
            "median_absolute_error": [172.543702],
            "r2_score": [0.87548],
            "explained_variance": [0.87548],
        },
        dtype="Float64",
    )
    expected = expected.reindex(index=expected.index.astype("Int64"))
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)

    # save, load, check parameters to ensure configuration was kept
    reloaded_model = model.to_gbq(f"{dataset_id}.temp_configured_model", replace=True)
    assert reloaded_model.fit_intercept is True
    assert reloaded_model.data_split_method == "NO_SPLIT"
