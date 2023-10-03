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

"""Private module: Helpers for I/O operations."""

import datetime
import textwrap
import types
from typing import Dict, Optional, Sequence, Union, Iterable

import google.cloud.bigquery as bigquery

IO_ORDERING_ID = "bqdf_row_nums"
MAX_LABELS_COUNT = 64


def create_job_configs_labels(
    job_configs_labels: Optional[Dict[str, str]],
    api_methods: Sequence[str],
) -> Dict[str, str]:
    # If there is no label set
    if job_configs_labels is None:
        labels = {}
        label_values = list(api_methods)
    else:
        labels = job_configs_labels.copy()
        cur_labels_len = len(job_configs_labels)
        api_methods_len = len(api_methods)
        # If the total number of labels is under the limit of labels count
        if cur_labels_len + api_methods_len <= MAX_LABELS_COUNT:
            label_values = list(api_methods)
        # We capture the latest label if it is out of the length limit of labels count
        else:
            added_api_len = cur_labels_len + api_methods_len - MAX_LABELS_COUNT
            label_values = list(api_methods)[-added_api_len:]

    for i, label_value in enumerate(label_values):
        if job_configs_labels is not None:
            label_key = "bigframes-api-" + str(i + len(job_configs_labels))
        else:
            label_key = "bigframes-api-" + str(i)
        labels[label_key] = label_value
    return labels


def create_export_csv_statement(
    table_id: str, uri: str, field_delimiter: str, header: bool
) -> str:
    return create_export_data_statement(
        table_id,
        uri,
        "CSV",
        {
            "field_delimiter": field_delimiter,
            "header": header,
        },
    )


def create_export_data_statement(
    table_id: str, uri: str, format: str, export_options: Dict[str, Union[bool, str]]
) -> str:
    all_options: Dict[str, Union[bool, str]] = {
        "uri": uri,
        "format": format,
        # TODO(swast): Does pandas have an option not to overwrite files?
        "overwrite": True,
    }
    all_options.update(export_options)
    export_options_str = ", ".join(
        format_option(key, value) for key, value in all_options.items()
    )
    # Manually generate ORDER BY statement since ibis will not always generate
    # it in the top level statement. This causes BigQuery to then run
    # non-distributed sort and run out of memory.
    return textwrap.dedent(
        f"""
        EXPORT DATA
        OPTIONS (
            {export_options_str}
        ) AS
        SELECT * EXCEPT ({IO_ORDERING_ID})
        FROM `{table_id}`
        ORDER BY {IO_ORDERING_ID}
        """
    )


def create_snapshot_sql(
    table_ref: bigquery.TableReference, current_timestamp: datetime.datetime
) -> str:
    """Query a table via 'time travel' for consistent reads."""

    # If we have a _SESSION table, assume that it's already a copy. Nothing to do here.
    if table_ref.dataset_id.upper() == "_SESSION":
        return f"SELECT * FROM `_SESSION`.`{table_ref.table_id}`"

    # If we have an anonymous query results table, it can't be modified and
    # there isn't any BigQuery time travel.
    if table_ref.dataset_id.startswith("_"):
        return f"SELECT * FROM `{table_ref.project}`.`{table_ref.dataset_id}`.`{table_ref.table_id}`"

    return textwrap.dedent(
        f"""
        SELECT *
        FROM `{table_ref.project}`.`{table_ref.dataset_id}`.`{table_ref.table_id}`
        FOR SYSTEM_TIME AS OF TIMESTAMP({repr(current_timestamp.isoformat())})
        """
    )


# BigQuery REST API returns types in Legacy SQL format
# https://cloud.google.com/bigquery/docs/data-types but we use Standard SQL
# names
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
BQ_STANDARD_TYPES = types.MappingProxyType(
    {
        "BOOLEAN": "BOOL",
        "INTEGER": "INT64",
        "FLOAT": "FLOAT64",
    }
)


def bq_field_to_type_sql(field: bigquery.SchemaField):
    if field.mode == "REPEATED":
        nested_type = bq_field_to_type_sql(
            bigquery.SchemaField(
                field.name, field.field_type, mode="NULLABLE", fields=field.fields
            )
        )
        return f"ARRAY<{nested_type}>"

    if field.field_type == "RECORD":
        nested_fields_sql = ", ".join(
            bq_field_to_sql(child_field) for child_field in field.fields
        )
        return f"STRUCT<{nested_fields_sql}>"

    type_ = field.field_type
    return BQ_STANDARD_TYPES.get(type_, type_)


def bq_field_to_sql(field: bigquery.SchemaField):
    name = field.name
    type_ = bq_field_to_type_sql(field)
    return f"`{name}` {type_}"


def bq_schema_to_sql(schema: Iterable[bigquery.SchemaField]):
    return ", ".join(bq_field_to_sql(field) for field in schema)


def format_option(key: str, value: Union[bool, str]) -> str:
    if isinstance(value, bool):
        return f"{key}=true" if value else f"{key}=false"
    return f"{key}={repr(value)}"
