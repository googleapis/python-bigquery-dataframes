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


def use_bigquery_dataframes(project_id: str):
    # [START bigquery_dataframes_set_options]
    import bigframes.pandas as bpd

    PROJECT_ID = project_id  # @param {type:"string"}
    REGION = "US"  # @param {type:"string"}

    # Set BigQuery DataFrames options
    bpd.options.bigquery.project = PROJECT_ID
    bpd.options.bigquery.location = REGION

    # [END bigquery_dataframes_set_options]

    # [START bigquery_dataframes_load_data_from_bigquery]
    # Create a DataFrame from a BigQuery table:
    query_or_table = "bigquery-public-data.ml_datasets.penguins"
    bq_df = bpd.read_gbq(query_or_table)
    # [END bigquery_dataframes_load_data_from_bigquery]

    # [START bigquery_dataframes_load_data_from_csv]
    filepath_or_buffer = (
        "gs://bigquery-public-data-ml-datasets/holidays_and_events_for_forecasting.csv"
    )
    df_from_gcs = bpd.read_csv(filepath_or_buffer)
    # Display the first few rows of the DataFrame:
    df_from_gcs.head()
    # [END bigquery_dataframes_load_data_from_csv]

    # [START bigquery_dataframes_inspect_and_manipulate_data]
    # Inspect one of the columns (or series) of the DataFrame:
    bq_df["body_mass_g"].head(10)

    # Compute the mean of this series:
    average_body_mass = bq_df["body_mass_g"].mean()
    print(f"average_body_mass: {average_body_mass}")

    # Calculate the mean body_mass_g by species using the groupby operation:
    bq_df["body_mass_g"].groupby(by=bq_df["species"]).mean().head()
    # [END bigquery_dataframes_inspect_and_manipulate_data]

    # TODO(ashleyxu): Add samples for loading DataFrames to BigQuery table.
