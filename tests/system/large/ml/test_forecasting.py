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

from bigframes.ml import forecasting


def test_arima_plus_model(time_series_df_default_index):
    model = forecasting.ARIMAPlus()
    train_X = time_series_df_default_index[["parsed_date"]]
    train_y = time_series_df_default_index[["total_visits"]]
    model.fit(train_X, train_y)

    # TODO(garrettwu): add tests save/load/eval tests
