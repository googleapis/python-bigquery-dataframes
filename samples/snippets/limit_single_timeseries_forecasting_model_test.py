# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (t
# you may not use this file except in compliance wi
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in
# distributed under the License is distributed on a
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eit
# See the License for the specific language governi
# limitations under the License.


def test_limit_single_timeseries(random_model_id: str) -> None:
    # your_model_id = random_model_id

    # [START bigquery_dataframes_bqml_limit_forecast_visualize]
    import bigframes.pandas as bpd

    df = bpd.read_gbq("bigquery-public-data.new_york.citibike_trips")

    features = bpd.DataFrame(
        {
            "num_trips": df.starttime,
            "date": df["starttime"].dt.date,
        }
    )
    date = df["starttime"].dt.date
    df.groupby([date])
    num_trips = features.groupby(["date"]).count()
    # [END bigquery_dataframes_bqml_limit_forecast_visualize]
    assert df is not None
    assert features is not None
    assert num_trips is not None
