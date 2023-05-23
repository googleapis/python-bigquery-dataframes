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

import typing

import pandas

import bigframes
import bigframes.ml.core


def test_model_eval(
    penguins_bqml_linear_model,
):
    result = penguins_bqml_linear_model.evaluate().compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [227.01223],
            "mean_squared_error": [81838.159892],
            "mean_squared_log_error": [0.00507],
            "median_absolute_error": [173.080816],
            "r2_score": [0.872377],
            "explained_variance": [0.872377],
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


def test_model_eval_with_data(penguins_bqml_linear_model, penguins_df_default_index):
    result = penguins_bqml_linear_model.evaluate(
        penguins_df_default_index.dropna()
    ).compute()
    expected = pandas.DataFrame(
        {
            "mean_absolute_error": [225.817334],
            "mean_squared_error": [80540.705944],
            "mean_squared_log_error": [0.004972],
            "median_absolute_error": [173.080816],
            "r2_score": [0.87529],
            "explained_variance": [0.87529],
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


def test_model_predict(
    penguins_bqml_linear_model: bigframes.ml.core.BqmlModel, new_penguins_df
):
    predictions = penguins_bqml_linear_model.predict(new_penguins_df).compute()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [4030.1, 3280.8, 3177.9]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]].sort_index(),
        expected,
        check_exact=False,
        rtol=1e-2,
    )


def test_model_predict_with_unnamed_index(
    penguins_bqml_linear_model: bigframes.ml.core.BqmlModel, new_penguins_df
):

    # This will result in an index that lacks a name, which the ML library will
    # need to persist through the call to ML.PREDICT
    new_penguins_df = new_penguins_df.reset_index()

    # remove the middle tag number to ensure we're really keeping the unnamed index
    new_penguins_df = typing.cast(
        bigframes.DataFrame, new_penguins_df[new_penguins_df.tag_number != 1672]
    )

    predictions = penguins_bqml_linear_model.predict(new_penguins_df).compute()

    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [4030.1, 3177.9]},
        dtype="Float64",
        index=pandas.Index([0, 2], dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]].sort_index(),
        expected,
        check_exact=False,
        rtol=1e-2,
    )
