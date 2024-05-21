# Copyright 2024 Google LLC
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

import numpy as np
import pandas as pd

from bigframes.ml import impute
import bigframes.pandas as bpd


def test_simple_imputer_normalized_fit_transform_default_params():
    missing_df = bpd.DataFrame(
        {
            "culmen_length_mm": [39.5, 38.5, 37.9],
            "culmen_depth_mm": [np.nan, 17.2, 18.1],
            "flipper_length_mm": [np.nan, 181.0, 188.0],
        }
    )
    imputer = impute.SimpleImputer(strategy="mean")
    result = imputer.fit_transform(
        missing_df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"]]
    ).to_pandas()

    expected = pd.DataFrame(
        {
            "imputer_culmen_length_mm": [39.5, 38.5, 37.9],
            "imputer_culmen_depth_mm": [17.65, 17.2, 18.1],
            "imputer_flipper_length_mm": [184.5, 181.0, 188.0],
        },
        dtype="Float64",
        index=pd.Index([0, 1, 2], dtype="Int64"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_simple_imputer_series_normalizes(new_penguins_df):
    missing_df = bpd.DataFrame(
        {
            "culmen_length_mm": [39.5, 38.5, 37.9],
            "culmen_depth_mm": [np.nan, 17.2, 18.1],
            "flipper_length_mm": [np.nan, 181.0, 188.0],
        }
    )
    imputer = impute.SimpleImputer()
    imputer.fit(missing_df["culmen_depth_mm"])

    result = imputer.transform(missing_df["culmen_depth_mm"]).to_pandas()
    result = imputer.transform(new_penguins_df).to_pandas()

    expected = pd.DataFrame(
        {
            "imputer_culmen_depth_mm": [18.8, 17.2, 18.1],
        },
        dtype="Float64",
        index=pd.Index([1633, 1672, 1690], name="tag_number", dtype="Int64"),
    )

    pd.testing.assert_frame_equal(result, expected, rtol=0.1)


def test_simple_imputer_save_load_mean(dataset_id):
    missing_df = bpd.DataFrame(
        {
            "culmen_length_mm": [39.5, 38.5, 37.9],
            "culmen_depth_mm": [np.nan, 17.2, 18.1],
            "flipper_length_mm": [np.nan, 181.0, 188.0],
        }
    )
    transformer = impute.SimpleImputer(strategy="mean")
    transformer.fit(
        missing_df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"]]
    )

    reloaded_transformer = transformer.to_gbq(
        f"{dataset_id}.temp_configured_model", replace=True
    )
    assert isinstance(reloaded_transformer, impute.SimpleImputer)
    assert reloaded_transformer.strategy == transformer.strategy
    assert reloaded_transformer._bqml_model is not None


def test_simple_imputer_save_load_most_frequent(dataset_id):
    missing_df = bpd.DataFrame(
        {
            "culmen_length_mm": [39.5, 38.5, 37.9],
            "culmen_depth_mm": [np.nan, 17.2, 18.1],
            "flipper_length_mm": [np.nan, 181.0, 188.0],
        }
    )
    transformer = impute.SimpleImputer(strategy="most_frequent")
    transformer.fit(
        missing_df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"]]
    )

    reloaded_transformer = transformer.to_gbq(
        f"{dataset_id}.temp_configured_model", replace=True
    )
    assert isinstance(reloaded_transformer, impute.SimpleImputer)
    assert reloaded_transformer.strategy == transformer.strategy
    assert reloaded_transformer._bqml_model is not None
