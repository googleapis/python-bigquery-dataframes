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


def run_quickstart(project_id: str):
    # [START bigframes_quickstart]
    import bigframes.pandas as pd

    # TODO: (Optional) Setup your session with the configuration. Some of these
    # settings cannot be changed once a session has started.
    pd.options.bigquery.project = "your-gcp-project-id"
    pd.options.bigquery.location = "us"
    # [END bigframes_quickstart]
    pd.options.bigquery.project = project_id
    # [START bigframes_quickstart]
    query_or_table = "bigquery-public-data.ml_datasets.penguins"
    df = pd.read_gbq(query_or_table)

    # Use the DataFrame just as you would a pandas DataFrame, but calculations
    # happen in the BigQuery query engine instead of the local system.
    average_body_mass = df["body_mass_g"].mean()

    print(f"average_body_mass: {average_body_mass}")

    # IMPORTANT: The `bigframes.pandas` package creates a BigQuery session for
    # queries and temporary tables. A BigQuery session has a limited lifetime
    # (https://cloud.google.com/bigquery/docs/sessions-intro#limitations) and
    # does not support concurrent queries. For long lived applications, create
    # session objects as needed, instead.

    import bigframes

    session_options = bigframes.BigQueryOptions()
    session_options.project = "your-gcp-project-id"
    session_options.location = "us"
    # [END bigframes_quickstart]
    session_options.project = project_id
    # [START bigframes_quickstart]
    session = bigframes.connect(session_options)
    df_session = session.read_gbq(query_or_table)
    average_body_mass = df_session["body_mass_g"].mean()
    print(f"average_body_mass (df_session): {average_body_mass}")
