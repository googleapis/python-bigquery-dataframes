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

import bigframes.ml.compose
import bigframes.ml.preprocessing


def test_columntransformer_init_expectedtransforms():
    onehot_transformer = bigframes.ml.preprocessing.OneHotEncoder()
    scaler_transformer = bigframes.ml.preprocessing.StandardScaler()
    column_transformer = bigframes.ml.compose.ColumnTransformer(
        [
            ("onehot", onehot_transformer, "species"),
            ("scale", scaler_transformer, ["culmen_length_mm", "flipper_length_mm"]),
        ]
    )

    assert column_transformer.transformers_ == [
        ("onehot", onehot_transformer, "species"),
        ("scale", scaler_transformer, "culmen_length_mm"),
        ("scale", scaler_transformer, "flipper_length_mm"),
    ]
