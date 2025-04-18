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

from __future__ import annotations

import math
import os
from typing import cast, Literal, Mapping, Optional, Sequence, Tuple, Union
import warnings
import weakref

import google.api_core.exceptions
from google.cloud import bigquery
import google.cloud.bigquery.job as bq_job
import google.cloud.bigquery.table as bq_table
import google.cloud.bigquery_storage_v1

import bigframes.core
import bigframes.core.compile
import bigframes.core.guid
import bigframes.core.nodes as nodes
import bigframes.core.ordering as order
import bigframes.core.tree_properties as tree_properties
import bigframes.dtypes
import bigframes.exceptions as bfe
import bigframes.features
from bigframes.session import executor, read_api_execution
import bigframes.session._io.bigquery as bq_io
import bigframes.session.metrics
import bigframes.session.planner
import bigframes.session.temporary_storage

# Max complexity that should be executed as a single query
QUERY_COMPLEXITY_LIMIT = 1e7
# Number of times to factor out subqueries before giving up.
MAX_SUBTREE_FACTORINGS = 5
_MAX_CLUSTER_COLUMNS = 4
MAX_SMALL_RESULT_BYTES = 10 * 1024 * 1024 * 1024  # 10G


class BigQueryCachingExecutor(executor.Executor):
    """Computes BigFrames values using BigQuery Engine.

    This executor can cache expressions. If those expressions are executed later, this session
    will re-use the pre-existing results from previous executions.

    This class is not thread-safe.
    """

    def __init__(
        self,
        bqclient: bigquery.Client,
        storage_manager: bigframes.session.temporary_storage.TemporaryStorageManager,
        bqstoragereadclient: google.cloud.bigquery_storage_v1.BigQueryReadClient,
        *,
        strictly_ordered: bool = True,
        metrics: Optional[bigframes.session.metrics.ExecutionMetrics] = None,
    ):
        self.bqclient = bqclient
        self.storage_manager = storage_manager
        self.compiler: bigframes.core.compile.SQLCompiler = (
            bigframes.core.compile.SQLCompiler()
        )
        self.strictly_ordered: bool = strictly_ordered
        self._cached_executions: weakref.WeakKeyDictionary[
            nodes.BigFrameNode, nodes.BigFrameNode
        ] = weakref.WeakKeyDictionary()
        self.metrics = metrics
        self.bqstoragereadclient = bqstoragereadclient
        # Simple left-to-right precedence for now
        self._semi_executors = (
            read_api_execution.ReadApiSemiExecutor(
                bqstoragereadclient=bqstoragereadclient,
                project=self.bqclient.project,
            ),
        )

    def to_sql(
        self,
        array_value: bigframes.core.ArrayValue,
        offset_column: Optional[str] = None,
        ordered: bool = False,
        enable_cache: bool = True,
    ) -> str:
        if offset_column:
            array_value, _ = array_value.promote_offsets()
        node = (
            self.replace_cached_subtrees(array_value.node)
            if enable_cache
            else array_value.node
        )
        return self.compiler.compile(node, ordered=ordered)

    def execute(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        ordered: bool = True,
        use_explicit_destination: Optional[bool] = None,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
    ) -> executor.ExecuteResult:
        if use_explicit_destination is None:
            use_explicit_destination = bigframes.options.bigquery.allow_large_results

        if bigframes.options.compute.enable_multi_query_execution:
            self._simplify_with_caching(array_value)

        plan = self.replace_cached_subtrees(array_value.node)
        # Use explicit destination to avoid 10GB limit of temporary table
        destination_table = (
            self.storage_manager.create_temp_table(
                array_value.schema.to_bigquery(), cluster_cols=[]
            )
            if use_explicit_destination
            else None
        )
        return self._execute_plan(
            plan,
            ordered=ordered,
            page_size=page_size,
            max_results=max_results,
            destination=destination_table,
        )

    def export_gbq(
        self,
        array_value: bigframes.core.ArrayValue,
        destination: bigquery.TableReference,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        cluster_cols: Sequence[str] = [],
    ):
        """
        Export the ArrayValue to an existing BigQuery table.
        """
        if bigframes.options.compute.enable_multi_query_execution:
            self._simplify_with_caching(array_value)

        dispositions = {
            "fail": bigquery.WriteDisposition.WRITE_EMPTY,
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "append": bigquery.WriteDisposition.WRITE_APPEND,
        }
        sql = self.to_sql(array_value, ordered=False)
        job_config = bigquery.QueryJobConfig(
            write_disposition=dispositions[if_exists],
            destination=destination,
            clustering_fields=cluster_cols if cluster_cols else None,
        )
        # TODO(swast): plumb through the api_name of the user-facing api that
        # caused this query.
        _, query_job = self._run_execute_query(
            sql=sql,
            job_config=job_config,
        )

        has_timedelta_col = any(
            t == bigframes.dtypes.TIMEDELTA_DTYPE for t in array_value.schema.dtypes
        )

        if if_exists != "append" and has_timedelta_col:
            # Only update schema if this is not modifying an existing table, and the
            # new table contains timedelta columns.
            table = self.bqclient.get_table(destination)
            table.schema = array_value.schema.to_bigquery()
            self.bqclient.update_table(table, ["schema"])

        return query_job

    def export_gcs(
        self,
        array_value: bigframes.core.ArrayValue,
        uri: str,
        format: Literal["json", "csv", "parquet"],
        export_options: Mapping[str, Union[bool, str]],
    ):
        query_job = self.execute(
            array_value,
            ordered=False,
            use_explicit_destination=True,
        ).query_job
        assert query_job is not None
        result_table = query_job.destination
        assert result_table is not None
        export_data_statement = bq_io.create_export_data_statement(
            f"{result_table.project}.{result_table.dataset_id}.{result_table.table_id}",
            uri=uri,
            format=format,
            export_options=dict(export_options),
        )

        bq_io.start_query_with_client(
            self.bqclient,
            export_data_statement,
            job_config=bigquery.QueryJobConfig(),
            api_name=f"dataframe-to_{format.lower()}",
            metrics=self.metrics,
        )
        return query_job

    def dry_run(
        self, array_value: bigframes.core.ArrayValue, ordered: bool = True
    ) -> bigquery.QueryJob:
        sql = self.to_sql(array_value, ordered=ordered)
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = self.bqclient.query(sql, job_config=job_config)
        return query_job

    def peek(
        self,
        array_value: bigframes.core.ArrayValue,
        n_rows: int,
        use_explicit_destination: Optional[bool] = None,
    ) -> executor.ExecuteResult:
        """
        A 'peek' efficiently accesses a small number of rows in the dataframe.
        """
        plan = self.replace_cached_subtrees(array_value.node)
        if not tree_properties.can_fast_peek(plan):
            msg = bfe.format_message("Peeking this value cannot be done efficiently.")
            warnings.warn(msg)
        if use_explicit_destination is None:
            use_explicit_destination = bigframes.options.bigquery.allow_large_results

        destination_table = (
            self.storage_manager.create_temp_table(
                array_value.schema.to_bigquery(), cluster_cols=[]
            )
            if use_explicit_destination
            else None
        )

        return self._execute_plan(
            plan, ordered=False, destination=destination_table, peek=n_rows
        )

    def head(
        self, array_value: bigframes.core.ArrayValue, n_rows: int
    ) -> executor.ExecuteResult:

        maybe_row_count = self._local_get_row_count(array_value)
        if (maybe_row_count is not None) and (maybe_row_count <= n_rows):
            return self.execute(array_value, ordered=True)

        if not self.strictly_ordered and not array_value.node.explicitly_ordered:
            # No user-provided ordering, so just get any N rows, its faster!
            return self.peek(array_value, n_rows)

        plan = self.replace_cached_subtrees(array_value.node)
        if not tree_properties.can_fast_head(plan):
            # If can't get head fast, we are going to need to execute the whole query
            # Will want to do this in a way such that the result is reusable, but the first
            # N values can be easily extracted.
            # This currently requires clustering on offsets.
            self._cache_with_offsets(array_value)
            # Get a new optimized plan after caching
            plan = self.replace_cached_subtrees(array_value.node)
            assert tree_properties.can_fast_head(plan)

        head_plan = generate_head_plan(plan, n_rows)
        return self._execute_plan(head_plan, ordered=True)

    def get_row_count(self, array_value: bigframes.core.ArrayValue) -> int:
        # TODO: Fold row count node in and use local execution
        count = self._local_get_row_count(array_value)
        if count is not None:
            return count
        else:
            row_count_plan = self.replace_cached_subtrees(
                generate_row_count_plan(array_value.node)
            )
            results = self._execute_plan(row_count_plan, ordered=True)
            pa_table = next(results.arrow_batches())
            pa_array = pa_table.column(0)
            return pa_array.tolist()[0]

    def cached(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        force: bool = False,
        use_session: bool = False,
        cluster_cols: Sequence[str] = (),
    ) -> None:
        """Write the block to a session table."""
        # use a heuristic for whether something needs to be cached
        if (not force) and self._is_trivially_executable(array_value):
            return
        if use_session:
            self._cache_with_session_awareness(array_value)
        else:
            self._cache_with_cluster_cols(array_value, cluster_cols=cluster_cols)

    def _local_get_row_count(
        self, array_value: bigframes.core.ArrayValue
    ) -> Optional[int]:
        # optimized plan has cache materializations which will have row count metadata
        # that is more likely to be usable than original leaf nodes.
        plan = self.replace_cached_subtrees(array_value.node)
        return tree_properties.row_count(plan)

    # Helpers
    def _run_execute_query(
        self,
        sql: str,
        job_config: Optional[bq_job.QueryJobConfig] = None,
        api_name: Optional[str] = None,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
        query_with_job: bool = True,
    ) -> Tuple[bq_table.RowIterator, Optional[bigquery.QueryJob]]:
        """
        Starts BigQuery query job and waits for results.
        """
        job_config = bq_job.QueryJobConfig() if job_config is None else job_config
        if bigframes.options.compute.maximum_bytes_billed is not None:
            job_config.maximum_bytes_billed = (
                bigframes.options.compute.maximum_bytes_billed
            )

        if not self.strictly_ordered:
            job_config.labels["bigframes-mode"] = "unordered"

        try:
            iterator, query_job = bq_io.start_query_with_client(
                self.bqclient,
                sql,
                job_config=job_config,
                api_name=api_name,
                max_results=max_results,
                page_size=page_size,
                metrics=self.metrics,
                query_with_job=query_with_job,
            )
            return iterator, query_job

        except google.api_core.exceptions.BadRequest as e:
            # Unfortunately, this error type does not have a separate error code or exception type
            if "Resources exceeded during query execution" in e.message:
                new_message = "Computation is too complex to execute as a single query. Try using DataFrame.cache() on intermediate results, or setting bigframes.options.compute.enable_multi_query_execution."
                raise bigframes.exceptions.QueryComplexityError(new_message) from e
            else:
                raise

    def replace_cached_subtrees(self, node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        return nodes.top_down(node, lambda x: self._cached_executions.get(x, x))

    def _is_trivially_executable(self, array_value: bigframes.core.ArrayValue):
        """
        Can the block be evaluated very cheaply?
        If True, the array_value probably is not worth caching.
        """
        # Once rewriting is available, will want to rewrite before
        # evaluating execution cost.
        return tree_properties.is_trivially_executable(
            self.replace_cached_subtrees(array_value.node)
        )

    def _cache_with_cluster_cols(
        self, array_value: bigframes.core.ArrayValue, cluster_cols: Sequence[str]
    ):
        """Executes the query and uses the resulting table to rewrite future executions."""

        sql, schema, ordering_info = self.compiler.compile_raw(
            self.replace_cached_subtrees(array_value.node)
        )
        tmp_table = self._sql_as_cached_temp_table(
            sql,
            schema,
            cluster_cols=bq_io.select_cluster_cols(schema, cluster_cols),
        )
        cached_replacement = array_value.as_cached(
            cache_table=self.bqclient.get_table(tmp_table),
            ordering=ordering_info,
        ).node
        self._cached_executions[array_value.node] = cached_replacement

    def _cache_with_offsets(self, array_value: bigframes.core.ArrayValue):
        """Executes the query and uses the resulting table to rewrite future executions."""
        offset_column = bigframes.core.guid.generate_guid("bigframes_offsets")
        w_offsets, offset_column = array_value.promote_offsets()
        sql = self.compiler.compile(
            self.replace_cached_subtrees(w_offsets.node), ordered=False
        )

        tmp_table = self._sql_as_cached_temp_table(
            sql,
            w_offsets.schema.to_bigquery(),
            cluster_cols=[offset_column],
        )
        cached_replacement = array_value.as_cached(
            cache_table=self.bqclient.get_table(tmp_table),
            ordering=order.TotalOrdering.from_offset_col(offset_column),
        ).node
        self._cached_executions[array_value.node] = cached_replacement

    def _cache_with_session_awareness(
        self,
        array_value: bigframes.core.ArrayValue,
    ) -> None:
        session_forest = [obj._block._expr.node for obj in array_value.session.objects]
        # These node types are cheap to re-compute
        target, cluster_cols = bigframes.session.planner.session_aware_cache_plan(
            array_value.node, list(session_forest)
        )
        cluster_cols_sql_names = [id.sql for id in cluster_cols]
        if len(cluster_cols) > 0:
            self._cache_with_cluster_cols(
                bigframes.core.ArrayValue(target), cluster_cols_sql_names
            )
        elif self.strictly_ordered:
            self._cache_with_offsets(bigframes.core.ArrayValue(target))
        else:
            self._cache_with_cluster_cols(bigframes.core.ArrayValue(target), [])

    def _simplify_with_caching(self, array_value: bigframes.core.ArrayValue):
        """Attempts to handle the complexity by caching duplicated subtrees and breaking the query into pieces."""
        # Apply existing caching first
        for _ in range(MAX_SUBTREE_FACTORINGS):
            node_with_cache = self.replace_cached_subtrees(array_value.node)
            if node_with_cache.planning_complexity < QUERY_COMPLEXITY_LIMIT:
                return

            did_cache = self._cache_most_complex_subtree(array_value.node)
            if not did_cache:
                return

    def _cache_most_complex_subtree(self, node: nodes.BigFrameNode) -> bool:
        # TODO: If query fails, retry with lower complexity limit
        selection = tree_properties.select_cache_target(
            node,
            min_complexity=(QUERY_COMPLEXITY_LIMIT / 500),
            max_complexity=QUERY_COMPLEXITY_LIMIT,
            cache=dict(self._cached_executions),
            # Heuristic: subtree_compleixty * (copies of subtree)^2
            heuristic=lambda complexity, count: math.log(complexity)
            + 2 * math.log(count),
        )
        if selection is None:
            # No good subtrees to cache, just return original tree
            return False

        self._cache_with_cluster_cols(bigframes.core.ArrayValue(selection), [])
        return True

    def _sql_as_cached_temp_table(
        self,
        sql: str,
        schema: Sequence[bigquery.SchemaField],
        cluster_cols: Sequence[str],
    ) -> bigquery.TableReference:
        assert len(cluster_cols) <= _MAX_CLUSTER_COLUMNS
        temp_table = self.storage_manager.create_temp_table(schema, cluster_cols)

        # TODO: Get default job config settings
        job_config = cast(
            bigquery.QueryJobConfig,
            bigquery.QueryJobConfig.from_api_repr({}),
        )
        job_config.destination = temp_table
        _, query_job = self._run_execute_query(
            sql,
            job_config=job_config,
            api_name="cached",
        )
        assert query_job is not None
        query_job.result()
        return query_job.destination

    def _validate_result_schema(
        self,
        array_value: bigframes.core.ArrayValue,
        bq_schema: list[bigquery.SchemaField],
    ):
        actual_schema = _sanitize(tuple(bq_schema))
        ibis_schema = bigframes.core.compile.test_only_ibis_inferred_schema(
            self.replace_cached_subtrees(array_value.node)
        ).to_bigquery()
        internal_schema = _sanitize(array_value.schema.to_bigquery())
        if not bigframes.features.PANDAS_VERSIONS.is_arrow_list_dtype_usable:
            return

        if internal_schema != actual_schema:
            raise ValueError(
                f"This error should only occur while testing. BigFrames internal schema: {internal_schema} does not match actual schema: {actual_schema}"
            )

        if ibis_schema != actual_schema:
            raise ValueError(
                f"This error should only occur while testing. Ibis schema: {ibis_schema} does not match actual schema: {actual_schema}"
            )

    def _execute_plan(
        self,
        plan: nodes.BigFrameNode,
        ordered: bool,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
        destination: Optional[bq_table.TableReference] = None,
        peek: Optional[int] = None,
    ):
        """Just execute whatever plan as is, without further caching or decomposition."""

        # First try to execute fast-paths
        # TODO: Allow page_size and max_results by rechunking/truncating results
        if (not page_size) and (not max_results) and (not destination) and (not peek):
            for semi_executor in self._semi_executors:
                maybe_result = semi_executor.execute(plan, ordered=ordered)
                if maybe_result:
                    return maybe_result

        # TODO(swast): plumb through the api_name of the user-facing api that
        # caused this query.
        job_config = bigquery.QueryJobConfig()
        # Use explicit destination to avoid 10GB limit of temporary table
        if destination is not None:
            job_config.destination = destination
        sql = self.compiler.compile(plan, ordered=ordered, limit=peek)
        iterator, query_job = self._run_execute_query(
            sql=sql,
            job_config=job_config,
            page_size=page_size,
            max_results=max_results,
            query_with_job=(destination is not None),
        )

        # Though we provide the read client, iterator may or may not use it based on what is efficient for the result
        def iterator_supplier():
            # Workaround issue fixed by: https://github.com/googleapis/python-bigquery/pull/2154
            if iterator._page_size is not None or iterator.max_results is not None:
                return iterator.to_arrow_iterable(bqstorage_client=None)
            else:
                return iterator.to_arrow_iterable(
                    bqstorage_client=self.bqstoragereadclient
                )

        if query_job:
            size_bytes = self.bqclient.get_table(query_job.destination).num_bytes
        else:
            size_bytes = None

        if size_bytes is not None and size_bytes >= MAX_SMALL_RESULT_BYTES:
            msg = bfe.format_message(
                "The query result size has exceeded 10 GB. In BigFrames 2.0 and "
                "later, you might need to manually set `allow_large_results=True` in "
                "the IO method or adjust the BigFrames option: "
                "`bigframes.options.bigquery.allow_large_results=True`."
            )
            warnings.warn(msg, FutureWarning)
        # Runs strict validations to ensure internal type predictions and ibis are completely in sync
        # Do not execute these validations outside of testing suite.
        if "PYTEST_CURRENT_TEST" in os.environ:
            self._validate_result_schema(
                bigframes.core.ArrayValue(plan), iterator.schema
            )

        return executor.ExecuteResult(
            arrow_batches=iterator_supplier,
            schema=plan.schema,
            query_job=query_job,
            total_bytes=size_bytes,
            total_rows=iterator.total_rows,
        )


def _sanitize(
    schema: Tuple[bigquery.SchemaField, ...]
) -> Tuple[bigquery.SchemaField, ...]:
    # Schema inferred from SQL strings and Ibis expressions contain only names, types and modes,
    # so we disregard other fields (e.g timedelta description for timedelta columns) for validations.
    return tuple(
        bigquery.SchemaField(
            f.name,
            f.field_type,
            f.mode,  # type:ignore
            fields=_sanitize(f.fields),
        )
        for f in schema
    )


def generate_head_plan(node: nodes.BigFrameNode, n: int):
    return nodes.SliceNode(node, start=None, stop=n)


def generate_row_count_plan(node: nodes.BigFrameNode):
    return nodes.RowCountNode(node)
