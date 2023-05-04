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
import bigframes.ml.pipeline
import bigframes.ml.preprocessing


def test_pipeline_fit_score_predict(session, penguins_df_default_index):
    pipeline = bigframes.ml.pipeline.Pipeline(
        [
            ("scale", bigframes.ml.preprocessing.StandardScaler()),
            ("linreg", bigframes.ml.linear_model.LinearRegression()),
        ]
    )

    df = penguins_df_default_index.dropna()
    train_X = df[
        [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    train_y = df[["body_mass_g"]]
    pipeline.fit(train_X, train_y)

    score_result = pipeline.score().to_pandas()

    # Check score to ensure the model was fitted
    score_expected = pandas.DataFrame(
        {
            "mean_absolute_error": [309.477334],
            "mean_squared_error": [152184.227218],
            "mean_squared_log_error": [0.009524],
            "median_absolute_error": [257.727777],
            "r2_score": [0.764356],
            "explained_variance": [0.764356],
        },
        dtype="Float64",
    )
    score_expected = score_expected.reindex(index=score_expected.index.astype("Int64"))

    pandas.testing.assert_frame_equal(
        score_result, score_expected, check_exact=False, rtol=1e-2
    )

    # score on all training data
    score_result = pipeline.score(train_X, train_y).to_pandas()
    pandas.testing.assert_frame_equal(
        score_result, score_expected, check_exact=False, rtol=1e-2
    )

    # predict new labels
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
    predictions = pipeline.predict(new_penguins).to_pandas()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [3968.8, 3176.3, 3545.2]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]], expected, check_exact=False, rtol=1e-2
    )
