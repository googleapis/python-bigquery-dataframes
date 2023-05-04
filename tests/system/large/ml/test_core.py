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

import bigframes.ml.core
import bigframes.ml.sql


def test_bqml_e2e(session, dataset_id, penguins_df_default_index):
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

    model = bigframes.ml.core.create_bqml_model(
        train_X, train_y, {"model_type": "linear_reg"}
    )

    # no data - report evaluation from the automatic data split
    evaluate_result = model.evaluate().compute()
    evaluate_expected = pandas.DataFrame(
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
    evaluate_expected = evaluate_expected.reindex(
        index=evaluate_expected.index.astype("Int64")
    )
    pandas.testing.assert_frame_equal(
        evaluate_result, evaluate_expected, check_exact=False, rtol=1e-2
    )

    # evaluate on all training data
    evaluate_result = model.evaluate(df).compute()
    pandas.testing.assert_frame_equal(
        evaluate_result, evaluate_expected, check_exact=False, rtol=1e-2
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
    predictions = model.predict(new_penguins).compute()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [4030.1, 3280.8, 3177.9]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]], expected, check_exact=False, rtol=1e-2
    )

    new_name = f"{dataset_id}.my_model"
    new_model = model.copy(new_name, True)
    assert new_model.model_name == new_name

    fetch_result = session.bqclient.get_model(new_name)
    assert fetch_result.model_type == "LINEAR_REGRESSION"


def test_bqml_manual_preprocessing_e2e(session, dataset_id, penguins_df_default_index):
    df = penguins_df_default_index.dropna()
    train_X = df[
        [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
        ]
    ]
    train_y = df[["body_mass_g"]]
    transforms = [
        bigframes.ml.sql.ml_standard_scaler(column, column)
        for column in train_X.columns.tolist()
    ]
    transforms.extend(train_y.columns.tolist())
    options = {"model_type": "linear_reg"}
    model = bigframes.ml.core.create_bqml_model(
        train_X, train_y, transforms=transforms, options=options
    )

    # no data - report evaluation from the automatic data split
    evaluate_result = model.evaluate().compute()
    evaluate_expected = pandas.DataFrame(
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
    evaluate_expected = evaluate_expected.reindex(
        index=evaluate_expected.index.astype("Int64")
    )

    pandas.testing.assert_frame_equal(
        evaluate_result, evaluate_expected, check_exact=False, rtol=1e-2
    )

    # evaluate on all training data
    evaluate_result = model.evaluate(df).compute()
    pandas.testing.assert_frame_equal(
        evaluate_result, evaluate_expected, check_exact=False, rtol=1e-2
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
    predictions = model.predict(new_penguins).compute()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [3968.8, 3176.3, 3545.2]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions[["predicted_body_mass_g"]], expected, check_exact=False, rtol=1e-2
    )

    new_name = f"{dataset_id}.my_model"
    new_model = model.copy(new_name, True)
    assert new_model.model_name == new_name

    fetch_result = session.bqclient.get_model(new_name)
    assert fetch_result.model_type == "LINEAR_REGRESSION"
