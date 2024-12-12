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
    grouped_pm25_df = (
        filtered_pm25_df.groupby("date_local")
        .agg({"arithmetic_mean": "mean"})
        .reset_index()
    )

    # Rename the columns
    grouped_pm25_df = grouped_pm25_df.rename(
        columns={"arithmetic_mean": "pm25", "date_local": "date"}
    )

    # Assign only the pm25 and date columns to the table
    pm25_df = grouped_pm25_df[["pm25", "date"]]

    # Load wind_speed data
    wind_speed_df = bpd.read_gbq(
        "bigquery-public-data.epa_historical_air_quality.wind_daily_summary"
    )

    # Filter for Seattle and the relevant parameter
    filtered_wind_df = wind_speed_df[
        (wind_speed_df["city_name"] == "Seattle")
        & (wind_speed_df["parameter_name"] == "Wind Speed - Resultant")
    ]
    # Group by date_local and calculate the average arithmetic_mean
    grouped_wind_df = (
        filtered_wind_df.groupby("date_local")
        .agg({"arithmetic_mean": "mean"})
        .reset_index()
    )

    # Rename the columns
    grouped_wind_df = grouped_wind_df.rename(
        columns={"arithmetic_mean": "pm25", "date_local": "date"}
    )

    # Assign only the pm25 and date columns to the table
    wind_speed_df = grouped_wind_df[["pm25", "date"]]

    # Load temperature data
    temperature_df = bpd.read_gbq(
        "bigquery-public-data.epa_historical_air_quality.temperature_daily_summary"
    )

    # Filter for Seattle and the relevant parameter
    filtered_temperature_df = temperature_df[
        (temperature_df["city_name"] == "Seattle")
        & (temperature_df["parameter_name"] == "Outdoor Temperature")
    ]
    # Group by date_local and calculate the average arithmetic_mean
    grouped_temperature_df = (
        filtered_temperature_df.groupby("date_local")
        .agg({"arithmetic_mean": "mean"})
        .reset_index()
    )

    # Rename the columns
    grouped_temperature_df = grouped_temperature_df.rename(
        columns={"arithmetic_mean": "pm25", "date_local": "date"}
    )

    # Assign only the pm25 and date columns to the table
    temperature_df = grouped_temperature_df[["pm25", "date"]]

    # Merge the tables
    seattle_air_quality_daily = pm25_df.merge(wind_speed_df, on="date").merge(
        temperature_df, on="date"
    )
    # [END bigquery_dataframes_anomaly_prepare]
    pm25_df is not None
    wind_speed_df is not None
    temperature_df is not None
    seattle_air_quality_daily is not None
