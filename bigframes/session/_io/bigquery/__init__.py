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

"""Private module: Helpers for BigQuery I/O operations."""

from __future__ import annotations

import datetime
import itertools
import os
import textwrap
import types
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import google.api_core.exceptions
import google.cloud.bigquery as bigquery

import bigframes
from bigframes.core import log_adapter
import bigframes.formatting_helpers as formatting_helpers

IO_ORDERING_ID = "bqdf_row_nums"
MAX_LABELS_COUNT = 64
_LIST_TABLES_LIMIT = 10000  # calls to bqclient.list_tables
# will be limited to this many tables

LOGGING_NAME_ENV_VAR = "BIGFRAMES_PERFORMANCE_LOG_NAME"


def create_job_configs_labels(
    job_configs_labels: Optional[Dict[str, str]],
    api_methods: Sequence[str],
) -> Dict[str, str]:
    if job_configs_labels is None:
        job_configs_labels = {}

    labels = list(
        itertools.chain(
            job_configs_labels.keys(),
            (f"recent-bigframes-api-{i}" for i in range(len(api_methods))),
        )
    )
    values = list(itertools.chain(job_configs_labels.values(), api_methods))
    return dict(zip(labels[:MAX_LABELS_COUNT], values[:MAX_LABELS_COUNT]))


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


def table_ref_to_sql(table: bigquery.TableReference) -> str:
    """Format a table reference as escaped SQL."""
    return f"`{table.project}`.`{table.dataset_id}`.`{table.table_id}`"


def create_temp_table(
    session: bigframes.session.Session,
    expiration: datetime.datetime,
    *,
    schema: Optional[Iterable[bigquery.SchemaField]] = None,
    cluster_columns: Optional[list[str]] = None,
) -> str:
    """Create an empty table with an expiration in the desired session.

    The table will be deleted when the session is closed or the expiration
    is reached.
    """
    bqclient: bigquery.Client = session.bqclient
    table_ref = session._random_table()
    destination = bigquery.Table(table_ref)
    destination.expires = expiration
    destination.schema = schema
    if cluster_columns:
        destination.clustering_fields = cluster_columns
    bqclient.create_table(destination)
    return f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"


def set_table_expiration(
    bqclient: bigquery.Client,
    table_ref: bigquery.TableReference,
    expiration: datetime.datetime,
) -> None:
    """Set an expiration time for an existing BigQuery table."""
    table = bqclient.get_table(table_ref)
    table.expires = expiration
    bqclient.update_table(table, ["expires"])


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


def start_query_with_client(
    bq_client: bigquery.Client,
    sql: str,
    job_config: bigquery.job.QueryJobConfig,
    max_results: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Tuple[bigquery.table.RowIterator, bigquery.QueryJob]:
    """
    Starts query job and waits for results.
    """
    api_methods = log_adapter.get_and_reset_api_methods()
    job_config.labels = create_job_configs_labels(
        job_configs_labels=job_config.labels, api_methods=api_methods
    )

    try:
        query_job = bq_client.query(sql, job_config=job_config, timeout=timeout)
    except google.api_core.exceptions.Forbidden as ex:
        if "Drive credentials" in ex.message:
            ex.message += "\nCheck https://cloud.google.com/bigquery/docs/query-drive-data#Google_Drive_permissions."
        raise

    opts = bigframes.options.display
    if opts.progress_bar is not None and not query_job.configuration.dry_run:
        results_iterator = formatting_helpers.wait_for_query_job(
            query_job, max_results, opts.progress_bar
        )
    else:
        results_iterator = query_job.result(max_results=max_results)

    if LOGGING_NAME_ENV_VAR in os.environ:
        # when running notebooks via pytest nbmake
        pytest_log_job(query_job)

    return results_iterator, query_job


def pytest_log_job(query_job: bigquery.QueryJob):
    """For pytest runs only, log information about the query job
    to a file in order to create a performance report.
    """
    if LOGGING_NAME_ENV_VAR not in os.environ:
        raise EnvironmentError(
            "Environment variable {env_var} is not set".format(
                env_var=LOGGING_NAME_ENV_VAR
            )
        )
    test_name = os.environ[LOGGING_NAME_ENV_VAR]
    current_directory = os.getcwd()
    bytes_processed = query_job.total_bytes_processed
    if not isinstance(bytes_processed, int):
        return  # filter out mocks
    if query_job.configuration.dry_run:
        # dry runs don't process their total_bytes_processed
        bytes_processed = 0
    bytes_file = os.path.join(current_directory, test_name + ".bytesprocessed")
    with open(bytes_file, "a") as f:
        f.write(str(bytes_processed) + "\n")


def delete_tables_matching_session_id(
    client: bigquery.Client, dataset: bigquery.DatasetReference, session_id: str
) -> None:
    """Searches within the dataset for tables conforming to the
    expected session_id form, and instructs bigquery to delete them.

    Args:
        client (bigquery.Client):
            The client to use to list tables
        dataset (bigquery.DatasetReference):
            The dataset to search in
        session_id (str):
            The session id to match on in the table name

    Returns:
        None
    """

    tables = client.list_tables(
        dataset, max_results=_LIST_TABLES_LIMIT, page_size=_LIST_TABLES_LIMIT
    )
    for table in tables:
        split_id = table.table_id.split("_")
        if not split_id[0].startswith("bqdf") or len(split_id) < 2:
            continue
        found_session_id = split_id[1]
        if found_session_id == session_id:
            client.delete_table(table, not_found_ok=True)
            print("Deleting temporary table '{}'.".format(table.table_id))


def create_bq_dataset_reference(
    bq_client: bigquery.Client, location=None, project=None
) -> bigquery.DatasetReference:
    """Create and identify dataset(s) for temporary BQ resources.

    bq_client project and location will be used unless kwargs "project"
    and/or "location" are given. If given, location and project
    will be passed through to
    https://cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client#google_cloud_bigquery_client_Client_query

    Args:
        bq_client (bigquery.Client):
            The bigquery.Client to use for the http request to
            create the dataset reference.
        location (str, default None):
            The location of the project to create the dataset in.
        project (str, default None):
            The project id of the project to create the dataset in.

    Returns:
        bigquery.DatasetReference: The constructed reference to the anonymous dataset.
    """
    query_job = bq_client.query("SELECT 1", location=location, project=project)
    query_job.result()  # blocks until finished

    # The anonymous dataset is used by BigQuery to write query results and
    # session tables. BigQuery DataFrames also writes temp tables directly
    # to the dataset, no BigQuery Session required. Note: there is a
    # different anonymous dataset per location. See:
    # https://cloud.google.com/bigquery/docs/cached-results#how_cached_results_are_stored
    query_destination = query_job.destination
    return bigquery.DatasetReference(
        query_destination.project,
        query_destination.dataset_id,
    )
