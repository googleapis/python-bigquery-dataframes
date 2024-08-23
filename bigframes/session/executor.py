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
from typing import cast, Iterable, Literal, Mapping, Optional, Sequence, Tuple
import warnings
import weakref

import google.api_core.exceptions
import google.cloud.bigquery as bigquery
import google.cloud.bigquery.job as bq_job

import bigframes.core
import bigframes.core.compile
import bigframes.core.guid
import bigframes.core.nodes as nodes
import bigframes.core.ordering as order
import bigframes.core.tree_properties as tree_properties
import bigframes.formatting_helpers as formatting_helpers
import bigframes.session._io.bigquery as bq_io
import bigframes.session.metrics
import bigframes.session.planner
import bigframes.session.temp_storage

# Max complexity that should be executed as a single query
QUERY_COMPLEXITY_LIMIT = 1e7
# Number of times to factor out subqueries before giving up.
MAX_SUBTREE_FACTORINGS = 5

_MAX_CLUSTER_COLUMNS = 4


class BigQueryCachingExecutor:
    """Computes BigFrames values using BigQuery Engine.

    This executor can cache expressions. If those expressions are executed later, this session
    will re-use the pre-existing results from previous executions.

    This class is not thread-safe.
    """

    def __init__(
        self,
        bqclient: bigquery.Client,
        storage_manager: bigframes.session.temp_storage.TemporaryGbqStorageManager,
        strictly_ordered: bool = True,
        metrics: Optional[bigframes.session.metrics.ExecutionMetrics] = None,
    ):
        self.bqclient = bqclient
        self.storage_manager = storage_manager
        self.compiler: bigframes.core.compile.SQLCompiler = (
            bigframes.core.compile.SQLCompiler(strict=strictly_ordered)
        )
        self.strictly_ordered: bool = strictly_ordered
        self._cached_executions: weakref.WeakKeyDictionary[
            nodes.BigFrameNode, nodes.BigFrameNode
        ] = weakref.WeakKeyDictionary()
        self.metrics = metrics

    def to_sql(
        self,
        array_value: bigframes.core.ArrayValue,
        offset_column: Optional[str] = None,
        col_id_overrides: Mapping[str, str] = {},
        ordered: bool = False,
        enable_cache: bool = True,
    ) -> str:
        """
        Convert an ArrayValue to a sql query that will yield its value.
        """
        if offset_column:
            array_value = array_value.promote_offsets(offset_column)
        node = (
            self._with_cached_executions(array_value.node)
            if enable_cache
            else array_value.node
        )
        if ordered:
            return self.compiler.compile_ordered(
                node, col_id_overrides=col_id_overrides
            )
        return self.compiler.compile_unordered(node, col_id_overrides=col_id_overrides)

    def execute(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        ordered: bool = True,
        col_id_overrides: Mapping[str, str] = {},
    ):
        """
        Execute the ArrayValue, storing the result to a temporary session-owned table.
        """
        if bigframes.options.compute.enable_multi_query_execution:
            self._simplify_with_caching(array_value)

        sql = self.to_sql(
            array_value, ordered=ordered, col_id_overrides=col_id_overrides
        )
        job_config = bigquery.QueryJobConfig()
        # TODO(swast): plumb through the api_name of the user-facing api that
        # caused this query.
        return self._run_execute_query(
            sql=sql,
            job_config=job_config,
        )

    def export(
        self,
        array_value,
        col_id_overrides: Mapping[str, str],
        destination: bigquery.TableReference,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        cluster_cols: Sequence[str] = [],
    ):
        """
        Export the ArrayValue to an existing BigQuery table.
        """
        dispositions = {
            "fail": bigquery.WriteDisposition.WRITE_EMPTY,
            "replace": bigquery.WriteDisposition.WRITE_TRUNCATE,
            "append": bigquery.WriteDisposition.WRITE_APPEND,
        }
        sql = self.to_sql(array_value, ordered=False, col_id_overrides=col_id_overrides)
        job_config = bigquery.QueryJobConfig(
            write_disposition=dispositions[if_exists],
            destination=destination,
            clustering_fields=cluster_cols if cluster_cols else None,
        )
        # TODO(swast): plumb through the api_name of the user-facing api that
        # caused this query.
        return self._run_execute_query(
            sql=sql,
            job_config=job_config,
        )

    def dry_run(self, array_value: bigframes.core.ArrayValue, ordered: bool = True):
        """
        Dry run executing the ArrayValue.

        Does not actually execute the data but will get stats and indicate any invalid query errors.
        """
        sql = self.to_sql(array_value, ordered=ordered)
        job_config = bigquery.QueryJobConfig(dry_run=True)
        bq_io.add_labels(job_config)
        query_job = self.bqclient.query(sql, job_config=job_config)
        results_iterator = query_job.result()
        return results_iterator, query_job

    def peek(
        self, array_value: bigframes.core.ArrayValue, n_rows: int
    ) -> tuple[bigquery.table.RowIterator, bigquery.QueryJob]:
        """A 'peek' efficiently accesses a small number of rows in the dataframe."""
        if not tree_properties.peekable(self._with_cached_executions(array_value.node)):
            warnings.warn("Peeking this value cannot be done efficiently.")
        sql = self.compiler.compile_peek(
            self._with_cached_executions(array_value.node), n_rows
        )

        # TODO(swast): plumb through the api_name of the user-facing api that
        # caused this query.
        return self._run_execute_query(
            sql=sql,
        )

    # Helpers
    def _run_execute_query(
        self,
        sql: str,
        job_config: Optional[bq_job.QueryJobConfig] = None,
        api_name: Optional[str] = None,
    ) -> Tuple[bigquery.table.RowIterator, bigquery.QueryJob]:
        """
        Starts BigQuery query job and waits for results.
        """
        job_config = bq_job.QueryJobConfig() if job_config is None else job_config
        if bigframes.options.compute.maximum_bytes_billed is not None:
            job_config.maximum_bytes_billed = (
                bigframes.options.compute.maximum_bytes_billed
            )
        # Note: add_labels is global scope which may have unexpected effects
        bq_io.add_labels(job_config, api_name=api_name)

        if not self.strictly_ordered:
            job_config.labels["bigframes-mode"] = "unordered"
        try:
            query_job = self.bqclient.query(sql, job_config=job_config)
            opts = bigframes.options.display
            if opts.progress_bar is not None and not query_job.configuration.dry_run:
                results_iterator = formatting_helpers.wait_for_query_job(
                    query_job, progress_bar=opts.progress_bar
                )
            else:
                results_iterator = query_job.result()

            if self.metrics is not None:
                self.metrics.count_job_stats(query_job)
            return results_iterator, query_job

        except google.api_core.exceptions.BadRequest as e:
            # Unfortunately, this error type does not have a separate error code or exception type
            if "Resources exceeded during query execution" in e.message:
                new_message = "Computation is too complex to execute as a single query. Try using DataFrame.cache() on intermediate results, or setting bigframes.options.compute.enable_multi_query_execution."
                raise bigframes.exceptions.QueryComplexityError(new_message) from e
            else:
                raise

    def _with_cached_executions(self, node: nodes.BigFrameNode) -> nodes.BigFrameNode:
        return tree_properties.replace_nodes(node, (dict(self._cached_executions)))

    def _is_trivially_executable(self, array_value: bigframes.core.ArrayValue):
        """
        Can the block be evaluated very cheaply?
        If True, the array_value probably is not worth caching.
        """
        # Once rewriting is available, will want to rewrite before
        # evaluating execution cost.
        return tree_properties.is_trivially_executable(
            self._with_cached_executions(array_value.node)
        )

    def _cache_with_cluster_cols(
        self, array_value: bigframes.core.ArrayValue, cluster_cols: Sequence[str]
    ):
        """Executes the query and uses the resulting table to rewrite future executions."""

        sql, schema, ordering_info = self.compiler.compile_raw(
            self._with_cached_executions(array_value.node)
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

        if not self.strictly_ordered:
            raise ValueError(
                "Caching with offsets only supported in strictly ordered mode."
            )
        offset_column = bigframes.core.guid.generate_guid("bigframes_offsets")
        node_w_offsets = array_value.promote_offsets(offset_column).node
        sql = self.compiler.compile_unordered(
            self._with_cached_executions(node_w_offsets)
        )

        tmp_table = self._sql_as_cached_temp_table(
            sql,
            node_w_offsets.schema.to_bigquery(),
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
        session_forest: Iterable[nodes.BigFrameNode],
    ) -> None:
        # These node types are cheap to re-compute
        target, cluster_cols = bigframes.session.planner.session_aware_cache_plan(
            array_value.node, list(session_forest)
        )
        if len(cluster_cols) > 0:
            self._cache_with_cluster_cols(
                bigframes.core.ArrayValue(target), cluster_cols
            )
        elif self.strictly_ordered:
            self._cache_with_offsets(bigframes.core.ArrayValue(target))
        else:
            self._cache_with_cluster_cols(bigframes.core.ArrayValue(target), [])

    def _simplify_with_caching(self, array_value: bigframes.core.ArrayValue):
        """Attempts to handle the complexity by caching duplicated subtrees and breaking the query into pieces."""
        # Apply existing caching first
        for _ in range(MAX_SUBTREE_FACTORINGS):
            node_with_cache = self._with_cached_executions(array_value.node)
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
        return query_job.destination
