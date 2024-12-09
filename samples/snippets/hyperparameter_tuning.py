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


def test_hyperparameter_tuning(random_model_id: str) -> None:
    # [START bigquery_dataframes_bqml_hyperparameter_table]
    import bigframes.pandas as bpd

    # Load data from BigQuery
    bq_df = bpd.read_gbq(
        "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2018"
    )

    # Drop rows with nulls to get training data
    training_data = bq_df.dropna(subset=["tip_amount"])
    bq_df.iloc[:10000]
    # [END bigquery_dataframes_bqml_hyperparameter_table]
    assert training_data is not None
