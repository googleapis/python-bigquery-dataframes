import math

import bigframes.ml.preprocessing


def test_standard_scaler_normalizes(penguins_df_default_index):
    # TODO(bmil): add a second test that compares output to sklearn.preprocessing.StandardScaler
    scaler = bigframes.ml.preprocessing.StandardScaler()
    scaler.fit(
        penguins_df_default_index[
            "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"
        ]
    )

    result = scaler.transform(
        penguins_df_default_index[
            "culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"
        ]
    ).to_pandas()
    for column in result.columns:
        assert math.isclose(result[column].mean(), 0.0, abs_tol=1e-9)
        assert math.isclose(result[column].std(), 1.0, abs_tol=1e-9)
