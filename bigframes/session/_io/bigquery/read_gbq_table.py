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

"""
Private helpers for loading a BigQuery table as a BigQuery DataFrames DataFrame.
"""

from __future__ import annotations

import datetime
import itertools
import os
import textwrap
import types
import typing
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import uuid
import warnings

import google.api_core.exceptions
import google.cloud.bigquery as bigquery
import ibis
import ibis.backends
import ibis.expr.types as ibis_types

import bigframes
import bigframes._config.bigquery_options as bigquery_options
import bigframes.clients
import bigframes.constants as constants
from bigframes.core import log_adapter
import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.compile
import bigframes.core.guid as guid
import bigframes.core.nodes as nodes
from bigframes.core.ordering import IntegerEncoding
import bigframes.core.ordering as order
import bigframes.core.tree_properties as traversals
import bigframes.core.tree_properties as tree_properties
import bigframes.core.utils as utils
import bigframes.dtypes
import bigframes.formatting_helpers as formatting_helpers
from bigframes.functions.remote_function import read_gbq_function as bigframes_rgf
from bigframes.functions.remote_function import remote_function as bigframes_rf
import bigframes.session._io.bigquery as bigframes_io
import bigframes.session._io.bigquery.read_gbq_table
import bigframes.session.clients
import bigframes.version

# Avoid circular imports.
if typing.TYPE_CHECKING:
    import bigframes.dataframe as dataframe


def get_table_metadata(
    bqclient: bigquery.Client,
    table_ref: google.cloud.bigquery.table.TableReference,
    *,
    api_name: str,
    cache: Dict[bigquery.TableReference, Tuple[datetime.datetime, bigquery.Table]],
    use_cache: bool = True,
) -> Tuple[datetime.datetime, google.cloud.bigquery.table.Table]:
    """Get the table metadata, either from cache or via REST API."""

    cached_table = cache.get(table_ref)
    if use_cache and cached_table is not None:
        snapshot_timestamp, _ = cached_table

        # Cache hit could be unexpected. See internal issue 329545805.
        # Raise a warning with more information about how to avoid the
        # problems with the cache.
        warnings.warn(
            f"Reading cached table from {snapshot_timestamp} to avoid "
            "incompatibilies with previous reads of this table. To read "
            "the latest version, set `use_cache=False` or close the "
            "current session with Session.close() or "
            "bigframes.pandas.close_session().",
            # There are many layers before we get to (possibly) the user's code:
            # pandas.read_gbq_table
            # -> with_default_session
            # -> Session.read_gbq_table
            # -> _read_gbq_table
            # -> _get_snapshot_sql_and_primary_key
            # -> get_snapshot_datetime_and_table_metadata
            stacklevel=7,
        )
        return cached_table

    # TODO(swast): It's possible that the table metadata is changed between now
    # and when we run the CURRENT_TIMESTAMP() query to see when we can time
    # travel to. Find a way to fetch the table metadata and BQ's current time
    # atomically.
    table = bqclient.get_table(table_ref)

    # TODO(b/336521938): Refactor to make sure we set the "bigframes-api"
    # whereever we execute a query.
    job_config = bigquery.QueryJobConfig()
    job_config.labels["bigframes-api"] = api_name
    snapshot_timestamp = list(
        bqclient.query(
            "SELECT CURRENT_TIMESTAMP() AS `current_timestamp`",
            job_config=job_config,
        ).result()
    )[0][0]
    cached_table = (snapshot_timestamp, table)
    cache[table_ref] = cached_table
    return cached_table


def _create_time_travel_sql(
    table_ref: bigquery.TableReference, time_travel_timestamp: datetime.datetime
) -> str:
    """Query a table via 'time travel' for consistent reads."""
    # If we have an anonymous query results table, it can't be modified and
    # there isn't any BigQuery time travel.
    if table_ref.dataset_id.startswith("_"):
        return f"SELECT * FROM `{table_ref.project}`.`{table_ref.dataset_id}`.`{table_ref.table_id}`"

    return textwrap.dedent(
        f"""
        SELECT *
        FROM `{table_ref.project}`.`{table_ref.dataset_id}`.`{table_ref.table_id}`
        FOR SYSTEM_TIME AS OF TIMESTAMP({repr(time_travel_timestamp.isoformat())})
        """
    )


def get_ibis_time_travel_table(
    ibis_client: ibis.BaseBackend,
    table_ref: bigquery.TableReference,
    time_travel_timestamp: datetime.datetime,
) -> ibis_types.Table:
    try:
        return ibis_client.sql(
            bigframes_io.create_snapshot_sql(table_ref, time_travel_timestamp)
        )
    except google.api_core.exceptions.Forbidden as ex:
        # Ibis does a dry run to get the types of the columns from the SQL.
        if "Drive credentials" in ex.message:
            ex.message += "\nCheck https://cloud.google.com/bigquery/docs/query-drive-data#Google_Drive_permissions."
        raise


def _check_index_uniqueness(
    self, table: ibis_types.Table, index_cols: List[str]
) -> bool:
    distinct_table = table.select(*index_cols).distinct()
    is_unique_sql = f"""WITH full_table AS (
        {self.ibis_client.compile(table)}
    ),
    distinct_table AS (
        {self.ibis_client.compile(distinct_table)}
    )

    SELECT (SELECT COUNT(*) FROM full_table) AS `total_count`,
    (SELECT COUNT(*) FROM distinct_table) AS `distinct_count`
    """
    results, _ = self._start_query(is_unique_sql)
    row = next(iter(results))

    total_count = row["total_count"]
    distinct_count = row["distinct_count"]
    return total_count == distinct_count


def get_index_and_maybe_total_ordering(
    table: bigquery.table.Table,
):
    """
    If we can get a total ordering from the table, such as via primary key
    column(s), then return those too so that ordering generation can be
    avoided.
    """
    # If there are primary keys defined, the query engine assumes these
    # columns are unique, even if the constraint is not enforced. We make
    # the same assumption and use these columns as the total ordering keys.
    primary_keys = None
    if (
        (table_constraints := getattr(table, "table_constraints", None)) is not None
        and (primary_key := table_constraints.primary_key) is not None
        # This will be False for either None or empty list.
        # We want primary_keys = None if no primary keys are set.
        and (columns := primary_key.columns)
    ):
        primary_keys = columns

    total_ordering_cols = primary_keys

    # TODO: warn if partitioned and/or clustered except if:
    # primary_keys, index_col, or filters
    # Except it looks like filters goes through the query path?

    if not index_col and primary_keys is not None:
        index_col = primary_keys

    if isinstance(index_col, str):
        index_cols = [index_col]
    else:
        index_cols = list(index_col)

    return index_cols, total_ordering_cols


def get_time_travel_datetime_and_table_metadata(
    bqclient: bigquery.Client,
    table_ref: bigquery.TableReference,
    *,
    api_name: str,
    cache: Dict[bigquery.TableReference, Tuple[datetime.datetime, bigquery.Table]],
    use_cache: bool = True,
) -> Tuple[datetime.datetime, bigquery.Table]:
    cached_table = cache.get(table_ref)
    if use_cache and cached_table is not None:
        snapshot_timestamp, _ = cached_table

        # Cache hit could be unexpected. See internal issue 329545805.
        # Raise a warning with more information about how to avoid the
        # problems with the cache.
        warnings.warn(
            f"Reading cached table from {snapshot_timestamp} to avoid "
            "incompatibilies with previous reads of this table. To read "
            "the latest version, set `use_cache=False` or close the "
            "current session with Session.close() or "
            "bigframes.pandas.close_session().",
            # There are many layers before we get to (possibly) the user's code:
            # pandas.read_gbq_table
            # -> with_default_session
            # -> Session.read_gbq_table
            # -> _read_gbq_table
            # -> _get_snapshot_sql_and_primary_key
            # -> get_snapshot_datetime_and_table_metadata
            stacklevel=7,
        )
        return cached_table

    # TODO(swast): It's possible that the table metadata is changed between now
    # and when we run the CURRENT_TIMESTAMP() query to see when we can time
    # travel to. Find a way to fetch the table metadata and BQ's current time
    # atomically.
    table = bqclient.get_table(table_ref)

    # TODO(b/336521938): Refactor to make sure we set the "bigframes-api"
    # whereever we execute a query.
    job_config = bigquery.QueryJobConfig()
    job_config.labels["bigframes-api"] = api_name
    snapshot_timestamp = list(
        bqclient.query(
            "SELECT CURRENT_TIMESTAMP() AS `current_timestamp`",
            job_config=job_config,
        ).result()
    )[0][0]
    cached_table = (snapshot_timestamp, table)
    cache[table_ref] = cached_table
    return cached_table


def to_ibis_table_with_time_travel(
    ibis_client: ibis.Backend,
    table_ref: bigquery.table.TableReference,
    snapshot_timestamp: datetime.datetime,
) -> Tuple[ibis_types.Table, Optional[Sequence[str]]]:
    """Create a read-only Ibis table expression representing a table."""
    try:
        table_expression = ibis_client.sql(
            create_snapshot_sql(table_ref, snapshot_timestamp)
        )
    except google.api_core.exceptions.Forbidden as ex:
        if "Drive credentials" in ex.message:
            ex.message += "\nCheck https://cloud.google.com/bigquery/docs/query-drive-data#Google_Drive_permissions."
        raise

    return table_expression


def to_array_value_with_total_ordering(
    self,
    table: ibis_types.Table,
) -> core.ArrayValue:
    # Since this might also be used as the index, don't use the default
    # "ordering ID" name.
    ordering_hash_part = guid.generate_guid("bigframes_ordering_")
    ordering_rand_part = guid.generate_guid("bigframes_ordering_")

    # All inputs into hash must be non-null or resulting hash will be null
    str_values = list(
        map(lambda col: _convert_to_nonnull_string(table[col]), table.columns)
    )
    full_row_str = (
        str_values[0].concat(*str_values[1:]) if len(str_values) > 1 else str_values[0]
    )
    full_row_hash = full_row_str.hash().name(ordering_hash_part)
    # Used to disambiguate between identical rows (which will have identical hash)
    random_value = ibis.random().name(ordering_rand_part)

    original_column_ids = table.columns
    table_with_ordering = table.select(
        itertools.chain(original_column_ids, [full_row_hash, random_value])
    )

    ordering_ref1 = order.ascending_over(ordering_hash_part)
    ordering_ref2 = order.ascending_over(ordering_rand_part)
    ordering = order.ExpressionOrdering(
        ordering_value_columns=(ordering_ref1, ordering_ref2),
        total_ordering_columns=frozenset([ordering_hash_part, ordering_rand_part]),
    )
    columns = [table_with_ordering[col] for col in original_column_ids]
    hidden_columns = [
        table_with_ordering[ordering_hash_part],
        table_with_ordering[ordering_rand_part],
    ]
    return core.ArrayValue.from_ibis(
        self,
        table_with_ordering,
        columns,
        hidden_ordering_columns=hidden_columns,
        ordering=ordering,
    )
