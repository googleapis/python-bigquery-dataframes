import google.api_core.exceptions
import pandas
import pytest

import bigframes.ml.linear_model


def test_model_eval(
    penguins_model_loaded: bigframes.ml.linear_model.LinearRegression,
):
    result = penguins_model_loaded.score().compute()
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
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)


def test_model_score_with_data(penguins_model_loaded, penguins_df_no_index):
    df = penguins_df_no_index.dropna()
    test_X = df[
        [
            "species",
            "island",
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "sex",
        ]
    ]
    test_y = df[["body_mass_g"]]
    result = penguins_model_loaded.score(test_X, test_y).compute()
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
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)


def test_model_predict(session, penguins_model_loaded):
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

    predictions = penguins_model_loaded.predict(new_penguins).compute()
    expected = pandas.DataFrame(
        {"predicted_body_mass_g": [4030.1, 3280.8, 3177.9]},
        dtype="Float64",
        index=pandas.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )
    pandas.testing.assert_frame_equal(
        predictions, expected, check_exact=False, rtol=1e-2
    )


def test_to_gbq_saved_model_scores(penguins_model_loaded, dataset_id):
    saved_model = penguins_model_loaded.to_gbq(
        f"{dataset_id}.test_penguins_model", replace=True
    )
    result = saved_model.score().compute()
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
    pandas.testing.assert_frame_equal(result, expected, check_exact=False, rtol=1e-2)


def test_to_gbq_replace(penguins_model_loaded, dataset_id):
    penguins_model_loaded.to_gbq(f"{dataset_id}.test_penguins_model", replace=True)
    with pytest.raises(google.api_core.exceptions.Conflict):
        penguins_model_loaded.to_gbq(f"{dataset_id}.test_penguins_model")
