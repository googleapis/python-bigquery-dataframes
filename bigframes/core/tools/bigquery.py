# Copyright 2025 Google LLC
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

"""
Private helpers for loading a BigQuery table as a BigQuery DataFrames DataFrame.
"""

from __future__ import annotations

import google.cloud.bigquery as bigquery


def get_schema_and_pseudocolumns(
    table: bigquery.table.Table,
) -> list[bigquery.SchemaField]:
    fields = list(table.schema)

    # TODO(tswast): Add _PARTITIONTIME and/or _PARTIONDATE for injestion
    # time partitioned tables.
    if table.table_id.endswith("*"):
        fields.append(
            bigquery.SchemaField(
                "_TABLE_SUFFIX",
                "STRING",
            )
        )

    return fields
