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


def test_create_single_timeseries(random_model_id):
    your_model_id = random_model_id

    # [START bigquery_dataframes_single_timeseries_forecasting_model_tutorial]
    import bigframes.pandas as bpd
    
    # Start by selecting the data that you'll be querying from bigquery-public-data.google_analytics_sample.ga_sessions_* 
    # The read_gbq function accepts the wildcard table expressions and this clause indicates that 
    # you are querying the ga_sessions_* tables in the google_analytics_sample dataset
    
    # Read and visualize the time series you want to forecast.
    df = bpd.read_gbq(
        'bigquery-public-data.google_analytics_sample.ga_sessions_*'
        )
    parsed_date = bpd.to_datetime(df.date, format= "%Y%m%d", utc = True)
    visits = df["totals"].struct.field("visits")
    total_visits = visits.groupby(parsed_date).sum()
    total_visits.plot.line()

    # [END bigquery_dataframes_single_timeseries_forecasting_model_tutorial]