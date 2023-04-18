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
    session: bigframes.Session, penguins_bqml_linear_model: bigframes.ml.core.BqmlModel
):
    new_penguins = session.read_pandas(
        pandas.DataFrame(
            {
                "tag_number": [1633, 1672, 1690],
                "species": [
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Adelie Penguin (Pygoscelis adeliae)",
                    "Chinstrap penguin (Pygoscelis antarctica)",
                ],
                "island": ["Torgersen", "Torgersen", "Dream"],
                "culmen_length_mm": [39.5, 38.5, 37.9],
                "culmen_depth_mm": [18.8, 17.2, 18.1],
                "flipper_length_mm": [196.0, 181.0, 188.0],
                "sex": ["MALE", "FEMALE", "FEMALE"],
            }
        ).set_index("tag_number")
    )

    predictions = penguins_bqml_linear_model.predict(new_penguins).compute()
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
