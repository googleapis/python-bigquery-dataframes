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
