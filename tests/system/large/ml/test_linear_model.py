import pandas

import bigframes.ml.linear_model


def test_linear_regression_configure_fit(penguins_df_default_index):
    model = bigframes.ml.linear_model.LinearRegression(fit_intercept=False)

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

    # save, load, check fit_intercept to ensure configuration was kept
    # the default value of fit_intercept is True, so this check ensures it was persisted
    # TODO(bmil): it looks like property is missing from the training options
    # https://cloud.google.com/bigquery/docs/reference/rest/v2/models#trainingoptions
    # reloaded_model = model.to_gbq(f"{dataset_id}.temp_configured_model", replace=True)
    # assert reloaded_model.fit_intercept == False
