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

import bigframes.ml.cluster
import bigframes.ml.linear_model
import bigframes.ml.pipeline
import bigframes.ml.preprocessing
from tests.system.utils import assert_pandas_df_equal_ignore_ordering


def test_pipeline_linreg_fit_score_predict(session, penguins_df_default_index):
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


def test_pipeline_kmeans_fit_predict(session, penguins_pandas_df_default_index):
    pipeline = bigframes.ml.pipeline.Pipeline(
        [
            ("scale", bigframes.ml.preprocessing.StandardScaler()),
            ("kmeans", bigframes.ml.cluster.KMeans(n_clusters=2)),
        ]
    )

    # kmeans is sensitive to the order with this configuration, so use ordered source data
    df = session.read_pandas(penguins_pandas_df_default_index).dropna()
    train_X = df[
        [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    pipeline.fit(train_X)

    # predict new labels
    pd_new_penguins = pandas.DataFrame.from_dict(
        {
            "test1": {
                "species": "Adelie Penguin (Pygoscelis adeliae)",
                "island": "Dream",
                "culmen_length_mm": 27.5,
                "culmen_depth_mm": 8.5,
                "flipper_length_mm": 99,
                "body_mass_g": 4475,
                "sex": "MALE",
            },
            "test2": {
                "species": "Chinstrap penguin (Pygoscelis antarctica)",
                "island": "Dream",
                "culmen_length_mm": 55.8,
                "culmen_depth_mm": 29.8,
                "flipper_length_mm": 307,
                "body_mass_g": 4000,
                "sex": "MALE",
            },
            "test3": {
                "species": "Adelie Penguin (Pygoscelis adeliae)",
                "island": "Biscoe",
                "culmen_length_mm": 19.7,
                "culmen_depth_mm": 8.9,
                "flipper_length_mm": 84,
                "body_mass_g": 3550,
                "sex": "MALE",
            },
            "test4": {
                "species": "Gentoo penguin (Pygoscelis papua)",
                "island": "Biscoe",
                "culmen_length_mm": 63.8,
                "culmen_depth_mm": 33.9,
                "flipper_length_mm": 298,
                "body_mass_g": 4300,
                "sex": "FEMALE",
            },
            "test5": {
                "species": "Adelie Penguin (Pygoscelis adeliae)",
                "island": "Dream",
                "culmen_length_mm": 27.5,
                "culmen_depth_mm": 8.5,
                "flipper_length_mm": 99,
                "body_mass_g": 4475,
                "sex": "MALE",
            },
            "test6": {
                "species": "Chinstrap penguin (Pygoscelis antarctica)",
                "island": "Dream",
                "culmen_length_mm": 55.8,
                "culmen_depth_mm": 29.8,
                "flipper_length_mm": 307,
                "body_mass_g": 4000,
                "sex": "MALE",
            },
        },
        orient="index",
    )
    pd_new_penguins.index.name = "observation"

    new_penguins = session.read_pandas(pd_new_penguins)
    result = pipeline.predict(new_penguins).to_pandas().sort_index()
    expected = pandas.DataFrame(
        {"CENTROID_ID": [1, 2, 1, 2, 1, 2]},
        dtype="Int64",
        index=pandas.Index(
            ["test1", "test2", "test3", "test4", "test5", "test6"],
            dtype="string[pyarrow]",
        ),
    )
    expected.index.name = "observation"
    assert_pandas_df_equal_ignore_ordering(result, expected)


def test_pipeline_onehotencoder_fit_predict(session, penguins_df_default_index):
    # TODO(bmil): right now this test covers basically the same behavior as the
    # StandardScaler case but with a different function. It should be reworked
    # into a test of ColumnComposer that uses both StandardScaler and OneHotEncoder
    pipeline = bigframes.ml.pipeline.Pipeline(
        [
            ("encode", bigframes.ml.preprocessing.OneHotEncoder()),
            ("linreg", bigframes.ml.linear_model.LinearRegression()),
        ]
    )

    df = penguins_df_default_index.dropna()
    train_X = df[["sex", "species", "island"]]
    train_y = df[["body_mass_g"]]
    pipeline.fit(train_X, train_y)

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
        {"predicted_body_mass_g": [4049.5, 3381.8, 3399.2]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]], expected, check_exact=False, rtol=1e-2
    )
