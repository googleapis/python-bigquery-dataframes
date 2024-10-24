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
from __future__ import annotations

from dataclasses import dataclass
import datetime
import functools
import io
import typing
from typing import Iterable, List, Optional, Sequence, Tuple
import warnings

import google.cloud.bigquery
import pandas
import pyarrow as pa
import pyarrow.feather as pa_feather

import bigframes.core.compile
import bigframes.core.expression as ex
import bigframes.core.guid
import bigframes.core.identifiers as ids
import bigframes.core.join_def as join_def
import bigframes.core.local_data as local_data
import bigframes.core.nodes as nodes
from bigframes.core.ordering import OrderingExpression
import bigframes.core.ordering as orderings
import bigframes.core.rewrite
import bigframes.core.schema as schemata
import bigframes.core.tree_properties
import bigframes.core.utils
from bigframes.core.window_spec import WindowSpec
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops
import bigframes.session._io.bigquery

if typing.TYPE_CHECKING:
    from bigframes.session import Session

ORDER_ID_COLUMN = "bigframes_ordering_id"
PREDICATE_COLUMN = "bigframes_predicate"


@dataclass(frozen=True)
class ArrayValue:
    """
    ArrayValue is an immutable type representing a 2D array with per-column types.
    """

    node: nodes.BigFrameNode

    @classmethod
    def from_pyarrow(cls, arrow_table: pa.Table, session: Session):
        adapted_table = local_data.adapt_pa_table(arrow_table)
        schema = local_data.arrow_schema_to_bigframes(adapted_table.schema)

        iobytes = io.BytesIO()
        pa_feather.write_feather(adapted_table, iobytes)
        # Scan all columns by default, we define this list as it can be pruned while preserving source_def
        scan_list = nodes.ScanList(
            tuple(
                nodes.ScanItem(ids.ColumnId(item.column), item.dtype, item.column)
                for item in schema.items
            )
        )

        node = nodes.ReadLocalNode(
            iobytes.getvalue(),
            data_schema=schema,
            session=session,
            n_rows=arrow_table.num_rows,
            scan_list=scan_list,
        )
        return cls(node)

    @classmethod
    def from_range(cls, start, end, step):
        return cls(
            nodes.FromRangeNode(
                start=start.node,
                end=end.node,
                step=step,
            )
        )

    @classmethod
    def from_table(
        cls,
        table: google.cloud.bigquery.Table,
        schema: schemata.ArraySchema,
        session: Session,
        *,
        predicate: Optional[str] = None,
        at_time: Optional[datetime.datetime] = None,
        primary_key: Sequence[str] = (),
        offsets_col: Optional[str] = None,
    ):
        if offsets_col and primary_key:
            raise ValueError("must set at most one of 'offests', 'primary_key'")
        if any(i.field_type == "JSON" for i in table.schema if i.name in schema.names):
            warnings.warn(
                "Interpreting JSON column(s) as StringDtype. This behavior may change in future versions.",
                bigframes.exceptions.PreviewWarning,
            )
        # define data source only for needed columns, this makes row-hashing cheaper
        table_def = nodes.GbqTable.from_table(table, columns=schema.names)

        # create ordering from info
        ordering = None
        if offsets_col:
            ordering = orderings.TotalOrdering.from_offset_col(offsets_col)
        elif primary_key:
            ordering = orderings.TotalOrdering.from_primary_key(primary_key)

        # Scan all columns by default, we define this list as it can be pruned while preserving source_def
        scan_list = nodes.ScanList(
            tuple(
                nodes.ScanItem(ids.ColumnId(item.column), item.dtype, item.column)
                for item in schema.items
            )
        )
        source_def = nodes.BigqueryDataSource(
            table=table_def, at_time=at_time, sql_predicate=predicate, ordering=ordering
        )
        node = nodes.ReadTableNode(
            source=source_def,
            scan_list=scan_list,
            table_session=session,
        )
        return cls(node)

    @property
    def column_ids(self) -> typing.Sequence[str]:
        """Returns column ids as strings."""
        return self.schema.names

    @property
    def session(self) -> Session:
        required_session = self.node.session
        from bigframes import get_global_session

        return (
            required_session if (required_session is not None) else get_global_session()
        )

    @functools.cached_property
    def schema(self) -> schemata.ArraySchema:
        return self.node.schema

    @property
    def explicitly_ordered(self) -> bool:
        # see BigFrameNode.explicitly_ordered
        return self.node.explicitly_ordered

    @property
    def order_ambiguous(self) -> bool:
        # see BigFrameNode.order_ambiguous
        return self.node.order_ambiguous

    @property
    def supports_fast_peek(self) -> bool:
        return bigframes.core.tree_properties.can_fast_peek(self.node)

    def as_cached(
        self: ArrayValue,
        cache_table: google.cloud.bigquery.Table,
        ordering: Optional[orderings.RowOrdering],
    ) -> ArrayValue:
        """
        Replace the node with an equivalent one that references a table where the value has been materialized to.
        """
        table = nodes.GbqTable.from_table(cache_table)
        source = nodes.BigqueryDataSource(table, ordering=ordering)
        # Assumption: GBQ cached table uses field name as bq column name
        scan_list = nodes.ScanList(
            tuple(
                nodes.ScanItem(field.id, field.dtype, field.id.name)
                for field in self.node.fields
            )
        )
        node = nodes.CachedTableNode(
            original_node=self.node,
            source=source,
            table_session=self.session,
            scan_list=scan_list,
        )
        return ArrayValue(node)

    def _try_evaluate_local(self):
        """Use only for unit testing paths - not fully featured. Will throw exception if fails."""
        return bigframes.core.compile.test_only_try_evaluate(self.node)

    def get_column_type(self, key: str) -> bigframes.dtypes.Dtype:
        return self.schema.get_type(key)

    def row_count(self) -> ArrayValue:
        """Get number of rows in ArrayValue as a single-entry ArrayValue."""
        return ArrayValue(nodes.RowCountNode(child=self.node))

    # Operations
    def filter_by_id(self, predicate_id: str, keep_null: bool = False) -> ArrayValue:
        """Filter the table on a given expression, the predicate must be a boolean series aligned with the table expression."""
        predicate: ex.Expression = ex.deref(predicate_id)
        if keep_null:
            predicate = ops.fillna_op.as_expr(predicate, ex.const(True))
        return self.filter(predicate)

    def filter(self, predicate: ex.Expression):
        return ArrayValue(nodes.FilterNode(child=self.node, predicate=predicate))

    def order_by(self, by: Sequence[OrderingExpression]) -> ArrayValue:
        return ArrayValue(nodes.OrderByNode(child=self.node, by=tuple(by)))

    def reversed(self) -> ArrayValue:
        return ArrayValue(nodes.ReversedNode(child=self.node))

    def slice(
        self, start: Optional[int], stop: Optional[int], step: Optional[int]
    ) -> ArrayValue:
        if self.node.order_ambiguous and not (self.session._strictly_ordered):
            warnings.warn(
                "Window ordering may be ambiguous, this can cause unstable results.",
                bigframes.exceptions.AmbiguousWindowWarning,
            )
        return ArrayValue(
            nodes.SliceNode(
                self.node,
                start=start,
                stop=stop,
                step=step if (step is not None) else 1,
            )
        )

    def promote_offsets(self) -> Tuple[ArrayValue, str]:
        """
        Convenience function to promote copy of column offsets to a value column. Can be used to reset index.
        """
        col_id = self._gen_namespaced_uid()
        if self.node.order_ambiguous and not (self.session._strictly_ordered):
            if not self.session._allows_ambiguity:
                raise ValueError(
                    "Generating offsets not supported in partial ordering mode"
                )
            else:
                warnings.warn(
                    "Window ordering may be ambiguous, this can cause unstable results.",
                    bigframes.exceptions.AmbiguousWindowWarning,
                )

        return (
            ArrayValue(
                nodes.PromoteOffsetsNode(child=self.node, col_id=ids.ColumnId(col_id))
            ),
            col_id,
        )

    def concat(self, other: typing.Sequence[ArrayValue]) -> ArrayValue:
        """Append together multiple ArrayValue objects."""
        return ArrayValue(
            nodes.ConcatNode(children=tuple([self.node, *[val.node for val in other]]))
        )

    def compute_values(self, assignments: Sequence[ex.Expression]):
        col_ids = self._gen_namespaced_uids(len(assignments))
        ex_id_pairs = tuple(
            (ex, ids.ColumnId(id)) for ex, id in zip(assignments, col_ids)
        )
        return (
            ArrayValue(nodes.ProjectionNode(child=self.node, assignments=ex_id_pairs)),
            col_ids,
        )

    def project_to_id(self, expression: ex.Expression):
        array_val, ids = self.compute_values(
            [expression],
        )
        return array_val, ids[0]

    def assign(self, source_id: str, destination_id: str) -> ArrayValue:
        if destination_id in self.column_ids:  # Mutate case
            exprs = [
                (
                    ex.deref(source_id if (col_id == destination_id) else col_id),
                    ids.ColumnId(col_id),
                )
                for col_id in self.column_ids
            ]
        else:  # append case
            self_projection = (
                (ex.deref(col_id), ids.ColumnId(col_id)) for col_id in self.column_ids
            )
            exprs = [
                *self_projection,
                (ex.deref(source_id), ids.ColumnId(destination_id)),
            ]
        return ArrayValue(
            nodes.SelectionNode(
                child=self.node,
                input_output_pairs=tuple(exprs),
            )
        )

    def create_constant(
        self,
        value: typing.Any,
        dtype: typing.Optional[bigframes.dtypes.Dtype],
    ) -> Tuple[ArrayValue, str]:
        if pandas.isna(value):
            # Need to assign a data type when value is NaN.
            dtype = dtype or bigframes.dtypes.DEFAULT_DTYPE

        return self.project_to_id(ex.const(value, dtype))

    def select_columns(self, column_ids: typing.Sequence[str]) -> ArrayValue:
        # This basically just drops and reorders columns - logically a no-op except as a final step
        selections = ((ex.deref(col_id), ids.ColumnId(col_id)) for col_id in column_ids)
        return ArrayValue(
            nodes.SelectionNode(
                child=self.node,
                input_output_pairs=tuple(selections),
            )
        )

    def drop_columns(self, columns: Iterable[str]) -> ArrayValue:
        return self.select_columns(
            [col_id for col_id in self.column_ids if col_id not in columns]
        )

    def aggregate(
        self,
        aggregations: typing.Sequence[typing.Tuple[ex.Aggregation, str]],
        by_column_ids: typing.Sequence[str] = (),
        dropna: bool = True,
    ) -> ArrayValue:
        """
        Apply aggregations to the expression.
        Arguments:
            aggregations: input_column_id, operation, output_column_id tuples
            by_column_id: column id of the aggregation key, this is preserved through the transform
            dropna: whether null keys should be dropped
        """
        agg_defs = tuple((agg, ids.ColumnId(name)) for agg, name in aggregations)
        return ArrayValue(
            nodes.AggregateNode(
                child=self.node,
                aggregations=agg_defs,
                by_column_ids=tuple(map(ex.deref, by_column_ids)),
                dropna=dropna,
            )
        )

    def project_window_op(
        self,
        column_name: str,
        op: agg_ops.UnaryWindowOp,
        window_spec: WindowSpec,
        *,
        never_skip_nulls=False,
        skip_reproject_unsafe: bool = False,
    ) -> Tuple[ArrayValue, str]:
        """
        Creates a new expression based on this expression with unary operation applied to one column.
        column_name: the id of the input column present in the expression
        op: the windowable operator to apply to the input column
        window_spec: a specification of the window over which to apply the operator
        output_name: the id to assign to the output of the operator, by default will replace input col if distinct output id not provided
        never_skip_nulls: will disable null skipping for operators that would otherwise do so
        skip_reproject_unsafe: skips the reprojection step, can be used when performing many non-dependent window operations, user responsible for not nesting window expressions, or using outputs as join, filter or aggregation keys before a reprojection
        """
        # TODO: Support non-deterministic windowing
        if window_spec.row_bounded or not op.order_independent:
            if self.node.order_ambiguous and not self.session._strictly_ordered:
                if not self.session._allows_ambiguity:
                    raise ValueError(
                        "Generating offsets not supported in partial ordering mode"
                    )
                else:
                    warnings.warn(
                        "Window ordering may be ambiguous, this can cause unstable results.",
                        bigframes.exceptions.AmbiguousWindowWarning,
                    )

        output_name = self._gen_namespaced_uid()
        return (
            ArrayValue(
                nodes.WindowOpNode(
                    child=self.node,
                    column_name=ex.deref(column_name),
                    op=op,
                    window_spec=window_spec,
                    output_name=ids.ColumnId(output_name),
                    never_skip_nulls=never_skip_nulls,
                    skip_reproject_unsafe=skip_reproject_unsafe,
                )
            ),
            output_name,
        )

    def relational_join(
        self,
        other: ArrayValue,
        conditions: typing.Tuple[typing.Tuple[str, str], ...] = (),
        type: typing.Literal["inner", "outer", "left", "right", "cross"] = "inner",
    ) -> typing.Tuple[ArrayValue, typing.Tuple[dict[str, str], dict[str, str]]]:
        l_mapping = {  # Identity mapping, only rename right side
            lcol.name: lcol.name for lcol in self.node.ids
        }
        r_mapping = {  # Rename conflicting names
            rcol.name: rcol.name
            if (rcol.name not in l_mapping)
            else bigframes.core.guid.generate_guid()
            for rcol in other.node.ids
        }
        other_node = other.node
        if set(other_node.ids) & set(self.node.ids):
            other_node = nodes.SelectionNode(
                other_node,
                tuple(
                    (ex.deref(old_id), ids.ColumnId(new_id))
                    for old_id, new_id in r_mapping.items()
                ),
            )

        join_node = nodes.JoinNode(
            left_child=self.node,
            right_child=other_node,
            conditions=tuple(
                (ex.deref(l_mapping[l_col]), ex.deref(r_mapping[r_col]))
                for l_col, r_col in conditions
            ),
            type=type,
        )
        return ArrayValue(join_node), (l_mapping, r_mapping)

    def try_align_as_projection(
        self,
        other: ArrayValue,
        join_type: join_def.JoinType,
        join_keys: typing.Tuple[join_def.CoalescedColumnMapping, ...],
        mappings: typing.Tuple[join_def.JoinColumnMapping, ...],
    ) -> typing.Optional[ArrayValue]:
        result = bigframes.core.rewrite.join_as_projection(
            self.node, other.node, join_keys, mappings, join_type
        )
        if result is not None:
            return ArrayValue(result)
        return None

    def explode(self, column_ids: typing.Sequence[str]) -> ArrayValue:
        assert len(column_ids) > 0
        for column_id in column_ids:
            assert bigframes.dtypes.is_array_like(self.get_column_type(column_id))

        offsets = tuple(ex.deref(id) for id in column_ids)
        return ArrayValue(nodes.ExplodeNode(child=self.node, column_ids=offsets))

    def _uniform_sampling(self, fraction: float) -> ArrayValue:
        """Sampling the table on given fraction.

        .. warning::
            The row numbers of result is non-deterministic, avoid to use.
        """
        return ArrayValue(nodes.RandomSampleNode(self.node, fraction))

    # Deterministically generate namespaced ids for new variables
    # These new ids are only unique within the current namespace.
    # Many operations, such as joins, create new namespaces. See: BigFrameNode.defines_namespace
    # When migrating to integer ids, these will generate the next available integer, in order to densely pack ids
    # this will help represent variables sets as compact bitsets
    def _gen_namespaced_uid(self) -> str:
        return self._gen_namespaced_uids(1)[0]

    def _gen_namespaced_uids(self, n: int) -> List[str]:
        i = len(self.node.defined_variables)
        genned_ids: List[str] = []
        while len(genned_ids) < n:
            attempted_id = f"col_{i}"
            if attempted_id not in self.node.defined_variables:
                genned_ids.append(attempted_id)
            i = i + 1
        return genned_ids
