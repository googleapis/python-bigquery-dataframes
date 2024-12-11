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


def test_multivariate_anomaly_detection(random_model_id: str) -> None:
    # your_model_id = random_model_id
    # [START bigquery_dataframes_anomaly_prepare]
    import bigframes.pandas as bpd

    # Load pm25_daily data
    pm25_df = bpd.read_gbq(
        "bigquery-public-data.epa_historical_air_quality.pm25_nonfrm_daily_summary"
    )

    # Filter for Seattle and the relevant parameter
    filtered_pm25_df = pm25_df[
        (pm25_df["city_name"] == "Seattle")
        & (pm25_df["parameter_name"] == "Acceptable PM2.5 AQI & Speciation Mass")
    ]
    # Group by date_local and calculate the average arithmetic_mean
    result_df = (
        filtered_pm25_df.groupby("date_local")
        .agg({"arithmetic_mean": "mean"})
        .reset_index()
    )

    # Rename the columns
    result_df = result_df.rename(
        columns={"arithmetic_mean": "pm25", "date_local": "date"}
    )

    # Assign only the pm25 and date columns to the table
    pm25_df = result_df[["pm25", "date"]]

    # Load wind_speed data
    wind_speed_df = bpd.read_gbq(
        "bigquery-public-data.epa_historical_air_quality.wind_daily_summary"
    )
    # Load temperature data
    temperature_df = bpd.read_gbq(
        "bigquery-public-data.epa_historical_air_quality.temperature_daily_summary"
    )

    # Merge the dataframes
    # seattle_air_quality_daily = pm25_daily.merge(wind_speed_df, on='date').merge(temperature_df, on='date')
    # Rename the 'date' column if necessary (optional)
    # seattle_air_quality_daily = seattle_air_quality_daily.rename(columns={"date": "date"})

    # Create the table (bpd.read_gbq(seattle_air_quality)?)

    # [END bigquery_dataframes_anomaly_prepare]
    pm25_df is not None
    wind_speed_df is not None
    temperature_df is not None
