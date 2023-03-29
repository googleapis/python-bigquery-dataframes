import pandas

import bigframes.ml.core


# TODO(bmil): add better test data for ML for realistic accuracy and prediction values
def test_bqml_e2e(session, dataset_id, scalars_dfs):
    scalars_df, _ = scalars_dfs
    scalars_df = scalars_df.dropna()
    train_X = scalars_df[["int64_col", "string_col"]]
    train_y = scalars_df[["float64_col"]]
    model = bigframes.ml.core.create_bqml_model(
        train_X, train_y, {"model_type": "linear_reg"}
    )

    # no data - report evaluation from the automatic data split
    evaluate_result = model.evaluate().compute()
    evaluate_expected = pandas.DataFrame(
        {
            "mean_absolute_error": [0.010732],
            "mean_squared_error": [0.0001455],
            "mean_squared_log_error": [0.000004862],
            "median_absolute_error": [0.012391],
            "r2_score": [1.0],
            "explained_variance": [1.0],
        }
    )

    # Just check that the columns are there, as with the very small number
    # of input rows the automatic training data split will cause a lot of
    # randomness in the values
    # TODO(bmil): use better test data, test this with tolerance instead
    pandas.testing.assert_index_equal(
        evaluate_result.columns,
        evaluate_expected.columns,
    )

    # evaluate on all training data
    # TODO(bmil): use better test data, test this with tolerance instead
    evaluate_result = model.evaluate(scalars_df).compute()
    pandas.testing.assert_index_equal(
        evaluate_result.columns,
        evaluate_expected.columns,
    )

    predict_result = model.predict(train_X).compute()
    predict_expected = pandas.DataFrame(
        {"predicted_float64_col": [2.52239, 1.25331, 24999999999.0]}, dtype="Float64"
    )
    if predict_result.index.name == "rowindex":
        predict_expected["rowindex"] = [1, 0, 2]
        predict_expected["rowindex"] = predict_expected["rowindex"].astype("Int64")
        predict_expected = predict_expected.set_index("rowindex")

    # Just check that the columns are there, as with the very small number
    # of input rows the automatic training data split will cause a lot of
    # randomness in the values
    # TODO(bmil): use better test data, test this with tolerance instead
    # assert_pandas_df_equal_ignore_ordering(predict_result, predict_expected, rtol=1e-3)
    pandas.testing.assert_index_equal(
        evaluate_result.columns,
        evaluate_expected.columns,
    )

    new_name = f"{dataset_id}.my_model"
    new_model = model.copy(new_name, True)
    assert new_model.model_name == new_name

    fetch_result = session.bqclient.get_model(new_name)
    assert fetch_result.model_type == "LINEAR_REGRESSION"
