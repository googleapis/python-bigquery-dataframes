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


def test_bigquery_dataframes_examples() -> None:
    # [START bigframes_bigquery_methods]
    import bigframes.bigquery as bbq
    import bigframes.pandas as bpd

    # Load data from BigQuery
    query_or_table = "bigquery-public-data.ml_datasets.penguins"
    bq_df = bpd.read_gbq(query_or_table)

    # Create a new STRUCT Series with subfields for each column in a DataFrames.
    lengths = bbq.struct(
        bq_df[["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm"]]
    )

    lengths.peek()

    # 146	{'culmen_length_mm': 51.1, 'culmen_depth_mm': ...
    # 278	{'culmen_length_mm': 48.2, 'culmen_depth_mm': ...
    # 337	{'culmen_length_mm': 36.4, 'culmen_depth_mm': ...
    # 154	{'culmen_length_mm': 46.5, 'culmen_depth_mm': ...
    # 185	{'culmen_length_mm': 50.1, 'culmen_depth_mm': ...
    # dtype: struct[pyarrow]
    # [END bigframes_bigquery_methods]
    assert lengths is not None
