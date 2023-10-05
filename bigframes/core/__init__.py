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
import functools
import math
import textwrap
import typing
from typing import Collection, Iterable, Literal, Optional, Sequence, Tuple

from google.cloud import bigquery
import ibis
import ibis.expr.datatypes as ibis_dtypes
import ibis.expr.types as ibis_types
import pandas

import bigframes.constants as constants
import bigframes.core.guid
from bigframes.core.ordering import (
    encode_order_string,
    ExpressionOrdering,
    IntegerEncoding,
    OrderingColumnReference,
    reencode_order_string,
    StringEncoding,
)
import bigframes.core.utils as utils
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops

if typing.TYPE_CHECKING:
    from bigframes.session import Session

ORDER_ID_COLUMN = "bigframes_ordering_id"
PREDICATE_COLUMN = "bigframes_predicate"


@dataclass(frozen=True)
class WindowSpec:
    """
    Specifies a window over which aggregate and analytic function may be applied.
    grouping_keys: set of column ids to group on
    preceding: Number of preceding rows in the window
    following: Number of preceding rows in the window
    ordering: List of columns ids and ordering direction to override base ordering
    """

    grouping_keys: typing.Sequence[str] = tuple()
    ordering: typing.Sequence[OrderingColumnReference] = tuple()
    preceding: typing.Optional[int] = None
    following: typing.Optional[int] = None
    min_periods: int = 0


# TODO(swast): We might want to move this to it's own sub-module.
class ArrayValue:
    """Immutable BigQuery DataFrames expression tree.

    Note: Usage of this class is considered to be private and subject to change
    at any time.

    This class is a wrapper around Ibis expressions. Its purpose is to defer
    Ibis projection operations to keep generated SQL small and correct when
    mixing and matching columns from different versions of a DataFrame.

    Args:
        session:
            A BigQuery DataFrames session to allow more flexibility in running
            queries.
        table: An Ibis table expression.
        columns: Ibis value expressions that can be projected as columns.
        hidden_ordering_columns: Ibis value expressions to store ordering.
        ordering: An ordering property of the data frame.
        predicates: A list of filters on the data frame.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        columns: Sequence[ibis_types.Value],
        hidden_ordering_columns: Optional[Sequence[ibis_types.Value]] = None,
        ordering: ExpressionOrdering = ExpressionOrdering(),
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self._session = session
        self._table = table
        self._predicates = tuple(predicates) if predicates is not None else ()
        # TODO: Validate ordering
        if not ordering.total_ordering_columns:
            raise ValueError("Must have total ordering defined by one or more columns")
        self._ordering = ordering
        # Allow creating a DataFrame directly from an Ibis table expression.
        # TODO(swast): Validate that each column references the same table (or
        # no table for literal values).
        self._columns = tuple(columns)

        # Meta columns store ordering, or other data that doesn't correspond to dataframe columns
        self._hidden_ordering_columns = (
            tuple(hidden_ordering_columns)
            if hidden_ordering_columns is not None
            else ()
        )

        # To allow for more efficient lookup by column name, create a
        # dictionary mapping names to column values.
        self._column_names = {column.get_name(): column for column in self._columns}
        self._hidden_ordering_column_names = {
            column.get_name(): column for column in self._hidden_ordering_columns
        }
        ### Validation
        value_col_ids = self._column_names.keys()
        hidden_col_ids = self._hidden_ordering_column_names.keys()

        all_columns = value_col_ids | hidden_col_ids
        ordering_valid = all(
            col.column_id in all_columns for col in ordering.all_ordering_columns
        )
        if value_col_ids & hidden_col_ids:
            raise ValueError(
                f"Keys in both hidden and exposed list: {value_col_ids & hidden_col_ids}"
            )
        if not ordering_valid:
            raise ValueError(f"Illegal ordering keys: {ordering.all_ordering_columns}")

    @classmethod
    def mem_expr_from_pandas(
        cls,
        pd_df: pandas.DataFrame,
        session: Optional[Session],
    ) -> ArrayValue:
        """
        Builds an in-memory only (SQL only) expr from a pandas dataframe.

        Caution: If session is None, only a subset of expr functionality will
        be available (null Session is usually not supported).
        """
        # We can't include any hidden columns in the ArrayValue constructor, so
        # grab the column names before we add the hidden ordering column.
        column_names = [str(column) for column in pd_df.columns]
        # Make sure column names are all strings.
        pd_df = pd_df.set_axis(column_names, axis="columns")
        pd_df = pd_df.assign(**{ORDER_ID_COLUMN: range(len(pd_df))})

        # ibis memtable cannot handle NA, must convert to None
        pd_df = pd_df.astype("object")  # type: ignore
        pd_df = pd_df.where(pandas.notnull(pd_df), None)

        # NULL type isn't valid in BigQuery, so retry with an explicit schema in these cases.
        keys_memtable = ibis.memtable(pd_df)
        schema = keys_memtable.schema()
        new_schema = []
        for column_index, column in enumerate(schema):
            if column == ORDER_ID_COLUMN:
                new_type: ibis_dtypes.DataType = ibis_dtypes.int64
            else:
                column_type = schema[column]
                # The autodetected type might not be one we can support, such
                # as NULL type for empty rows, so convert to a type we do
                # support.
                new_type = bigframes.dtypes.bigframes_dtype_to_ibis_dtype(
                    bigframes.dtypes.ibis_dtype_to_bigframes_dtype(column_type)
                )
                # TODO(swast): Ibis memtable doesn't use backticks in struct
                # field names, so spaces and other characters aren't allowed in
                # the memtable context. Blocked by
                # https://github.com/ibis-project/ibis/issues/7187
                column = f"col_{column_index}"
            new_schema.append((column, new_type))

        # must set non-null column labels. these are not the user-facing labels
        pd_df = pd_df.set_axis(
            [column for column, _ in new_schema],
            axis="columns",
        )
        keys_memtable = ibis.memtable(pd_df, schema=ibis.schema(new_schema))

        return cls(
            session,  # type: ignore # Session cannot normally be none, see "caution" above
            keys_memtable,
            columns=[
                keys_memtable[f"col_{column_index}"].name(column)
                for column_index, column in enumerate(column_names)
            ],
            ordering=ExpressionOrdering(
                ordering_value_columns=[OrderingColumnReference(ORDER_ID_COLUMN)],
                total_ordering_columns=frozenset([ORDER_ID_COLUMN]),
            ),
            hidden_ordering_columns=(keys_memtable[ORDER_ID_COLUMN],),
        )

    @property
    def columns(self) -> typing.Tuple[ibis_types.Value, ...]:
        return self._columns

    @property
    def column_ids(self) -> typing.Sequence[str]:
        return tuple(self._column_names.keys())

    @property
    def hidden_ordering_columns(self) -> typing.Tuple[ibis_types.Value, ...]:
        return self._hidden_ordering_columns

    @property
    def _reduced_predicate(self) -> typing.Optional[ibis_types.BooleanValue]:
        """Returns the frame's predicates as an equivalent boolean value, useful where a single predicate value is preferred."""
        return (
            _reduce_predicate_list(self._predicates).name(PREDICATE_COLUMN)
            if self._predicates
            else None
        )

    @property
    def _ibis_order(self) -> Sequence[ibis_types.Value]:
        """Returns a sequence of ibis values which can be directly used to order a table expression. Has direction modifiers applied."""
        return _convert_ordering_to_table_values(
            {**self._column_names, **self._hidden_ordering_column_names},
            self._ordering.all_ordering_columns,
        )

    def builder(self) -> ArrayValueBuilder:
        """Creates a mutable builder for expressions."""
        # Since ArrayValue is intended to be immutable (immutability offers
        # potential opportunities for caching, though we might need to introduce
        # more node types for that to be useful), we create a builder class.
        return ArrayValueBuilder(
            self._session,
            self._table,
            columns=self._columns,
            hidden_ordering_columns=self._hidden_ordering_columns,
            ordering=self._ordering,
            predicates=self._predicates,
        )

    def drop_columns(self, columns: Iterable[str]) -> ArrayValue:
        # Must generate offsets if we are dropping a column that ordering depends on
        expr = self
        for ordering_column in set(columns).intersection(
            [col.column_id for col in self._ordering.ordering_value_columns]
        ):
            expr = self._hide_column(ordering_column)

        expr_builder = expr.builder()
        remain_cols = [
            column for column in expr.columns if column.get_name() not in columns
        ]
        expr_builder.columns = remain_cols
        return expr_builder.build()

    def get_column_type(self, key: str) -> bigframes.dtypes.Dtype:
        ibis_type = typing.cast(
            bigframes.dtypes.IbisDtype, self._get_any_column(key).type()
        )
        return typing.cast(
            bigframes.dtypes.Dtype,
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_type),
        )

    def _get_ibis_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given column."""
        if key not in self.column_ids:
            raise ValueError(
                "Column name {} not in set of values: {}".format(key, self.column_ids)
            )
        return typing.cast(ibis_types.Value, self._column_names[key])

    def _get_any_column(self, key: str) -> ibis_types.Value:
        """Gets the Ibis expression for a given column. Will also get hidden columns."""
        all_columns = {**self._column_names, **self._hidden_ordering_column_names}
        if key not in all_columns.keys():
            raise ValueError(
                "Column name {} not in set of values: {}".format(
                    key, all_columns.keys()
                )
            )
        return typing.cast(ibis_types.Value, all_columns[key])

    def _get_hidden_ordering_column(self, key: str) -> ibis_types.Column:
        """Gets the Ibis expression for a given hidden column."""
        if key not in self._hidden_ordering_column_names.keys():
            raise ValueError(
                "Column name {} not in set of values: {}".format(
                    key, self._hidden_ordering_column_names.keys()
                )
            )
        return typing.cast(ibis_types.Column, self._hidden_ordering_column_names[key])

    def filter(self, predicate_id: str, keep_null: bool = False) -> ArrayValue:
        """Filter the table on a given expression, the predicate must be a boolean series aligned with the table expression."""
        condition = typing.cast(
            ibis_types.BooleanValue, self._get_ibis_column(predicate_id)
        )
        if keep_null:
            condition = typing.cast(
                ibis_types.BooleanValue,
                condition.fillna(
                    typing.cast(ibis_types.BooleanScalar, ibis_types.literal(True))
                ),
            )
        return self._filter(condition)

    def _filter(self, predicate_value: ibis_types.BooleanValue) -> ArrayValue:
        """Filter the table on a given expression, the predicate must be a boolean series aligned with the table expression."""
        expr = self.builder()
        expr.ordering = expr.ordering.with_non_sequential()
        expr.predicates = [*self._predicates, predicate_value]
        return expr.build()

    def order_by(
        self, by: Sequence[OrderingColumnReference], stable: bool = False
    ) -> ArrayValue:
        expr_builder = self.builder()
        expr_builder.ordering = self._ordering.with_ordering_columns(by, stable=stable)
        return expr_builder.build()

    def reversed(self) -> ArrayValue:
        expr_builder = self.builder()
        expr_builder.ordering = self._ordering.with_reverse()
        return expr_builder.build()

    def _uniform_sampling(self, fraction: float) -> ArrayValue:
        """Sampling the table on given fraction.

        .. warning::
            The row numbers of result is non-deterministic, avoid to use.
        """
        table = self._to_ibis_expr(
            "unordered", expose_hidden_cols=True, fraction=fraction
        )
        columns = [table[column_name] for column_name in self._column_names]
        hidden_ordering_columns = [
            table[column_name] for column_name in self._hidden_ordering_column_names
        ]
        return ArrayValue(
            self._session,
            table,
            columns=columns,
            hidden_ordering_columns=hidden_ordering_columns,
            ordering=self._ordering,
        )

    @property
    def _offsets(self) -> ibis_types.IntegerColumn:
        if not self._ordering.is_sequential:
            raise ValueError(
                "Expression does not have offsets. Generate them first using project_offsets."
            )
        if not self._ordering.total_order_col:
            raise ValueError(
                "Ordering is invalid. Marked as sequential but no total order columns."
            )
        column = self._get_any_column(self._ordering.total_order_col.column_id)
        return typing.cast(ibis_types.IntegerColumn, column)

    def _project_offsets(self) -> ArrayValue:
        """Create a new expression that contains offsets. Should only be executed when offsets are needed for an operations. Has no effect on expression semantics."""
        if self._ordering.is_sequential:
            return self
        # TODO(tbergeron): Enforce total ordering
        table = self._to_ibis_expr(
            ordering_mode="offset_col", order_col_name=ORDER_ID_COLUMN
        )
        columns = [table[column_name] for column_name in self._column_names]
        ordering = ExpressionOrdering(
            ordering_value_columns=[OrderingColumnReference(ORDER_ID_COLUMN)],
            total_ordering_columns=frozenset([ORDER_ID_COLUMN]),
            integer_encoding=IntegerEncoding(True, is_sequential=True),
        )
        return ArrayValue(
            self._session,
            table,
            columns=columns,
            hidden_ordering_columns=[table[ORDER_ID_COLUMN]],
            ordering=ordering,
        )

    def _hide_column(self, column_id) -> ArrayValue:
        """Pushes columns to hidden columns list. Used to hide ordering columns that have been dropped or destructively mutated."""
        expr_builder = self.builder()
        # Need to rename column as caller might be creating a new row with the same name but different values.
        # Can avoid this if don't allow callers to determine ids and instead generate unique ones in this class.
        new_name = bigframes.core.guid.generate_guid(prefix="bigframes_hidden_")
        expr_builder.hidden_ordering_columns = [
            *self._hidden_ordering_columns,
            self._get_ibis_column(column_id).name(new_name),
        ]
        expr_builder.ordering = self._ordering.with_column_remap({column_id: new_name})
        return expr_builder.build()

    def promote_offsets(self) -> typing.Tuple[ArrayValue, str]:
        """
        Convenience function to promote copy of column offsets to a value column. Can be used to reset index.
        """
        # Special case: offsets already exist
        ordering = self._ordering

        if (not ordering.is_sequential) or (not ordering.total_order_col):
            return self._project_offsets().promote_offsets()
        col_id = bigframes.core.guid.generate_guid()
        expr_builder = self.builder()
        expr_builder.columns = [
            self._get_any_column(ordering.total_order_col.column_id).name(col_id),
            *self.columns,
        ]
        return expr_builder.build(), col_id

    def select_columns(self, column_ids: typing.Sequence[str]):
        return self._projection(
            [self._get_ibis_column(col_id) for col_id in column_ids]
        )

    def _projection(self, columns: Iterable[ibis_types.Value]) -> ArrayValue:
        """Creates a new expression based on this expression with new columns."""
        # TODO(swast): We might want to do validation here that columns derive
        # from the same table expression instead of (in addition to?) at
        # construction time.

        expr = self
        for ordering_column in set(self.column_ids).intersection(
            [col_ref.column_id for col_ref in self._ordering.ordering_value_columns]
        ):
            # Need to hide ordering columns that are being dropped. Alternatively, could project offsets
            expr = expr._hide_column(ordering_column)
        builder = expr.builder()
        builder.columns = list(columns)
        new_expr = builder.build()
        return new_expr

    def shape(self) -> typing.Tuple[int, int]:
        """Returns dimensions as (length, width) tuple."""
        width = len(self.columns)
        count_expr = self._to_ibis_expr("unordered").count()
        sql = self._session.ibis_client.compile(count_expr)

        # Support in-memory engines for hermetic unit tests.
        if not isinstance(sql, str):
            length = self._session.ibis_client.execute(count_expr)
        else:
            row_iterator, _ = self._session._start_query(
                sql=sql,
                max_results=1,
            )
            length = next(row_iterator)[0]
        return (length, width)

    def concat(self, other: typing.Sequence[ArrayValue]) -> ArrayValue:
        """Append together multiple ArrayValue objects."""
        if len(other) == 0:
            return self
        tables = []
        prefix_base = 10
        prefix_size = math.ceil(math.log(len(other) + 1, prefix_base))
        # Must normalize all ids to the same encoding size
        max_encoding_size = max(
            self._ordering.string_encoding.length,
            *[expression._ordering.string_encoding.length for expression in other],
        )
        for i, expr in enumerate([self, *other]):
            ordering_prefix = str(i).zfill(prefix_size)
            table = expr._to_ibis_expr(
                ordering_mode="string_encoded", order_col_name=ORDER_ID_COLUMN
            )
            # Rename the value columns based on horizontal offset before applying union.
            table = table.select(
                [
                    table[col].name(f"column_{i}")
                    if col != ORDER_ID_COLUMN
                    else (
                        ordering_prefix
                        + reencode_order_string(
                            table[ORDER_ID_COLUMN], max_encoding_size
                        )
                    ).name(ORDER_ID_COLUMN)
                    for i, col in enumerate(table.columns)
                ]
            )
            tables.append(table)
        combined_table = ibis.union(*tables)
        ordering = ExpressionOrdering(
            ordering_value_columns=[OrderingColumnReference(ORDER_ID_COLUMN)],
            total_ordering_columns=frozenset([ORDER_ID_COLUMN]),
            string_encoding=StringEncoding(True, prefix_size + max_encoding_size),
        )
        return ArrayValue(
            self._session,
            combined_table,
            columns=[
                combined_table[col]
                for col in combined_table.columns
                if col != ORDER_ID_COLUMN
            ],
            hidden_ordering_columns=[combined_table[ORDER_ID_COLUMN]],
            ordering=ordering,
        )

    def project_unary_op(
        self, column_name: str, op: ops.UnaryOp, output_name=None
    ) -> ArrayValue:
        """Creates a new expression based on this expression with unary operation applied to one column."""
        value = op._as_ibis(self._get_ibis_column(column_name)).name(
            output_name or column_name
        )
        return self._set_or_replace_by_id(output_name or column_name, value)

    def project_binary_op(
        self,
        left_column_id: str,
        right_column_id: str,
        op: ops.BinaryOp,
        output_column_id: str,
    ) -> ArrayValue:
        """Creates a new expression based on this expression with binary operation applied to two columns."""
        value = op(
            self._get_ibis_column(left_column_id),
            self._get_ibis_column(right_column_id),
        ).name(output_column_id)
        return self._set_or_replace_by_id(output_column_id, value)

    def project_ternary_op(
        self,
        col_id_1: str,
        col_id_2: str,
        col_id_3: str,
        op: ops.TernaryOp,
        output_column_id: str,
    ) -> ArrayValue:
        """Creates a new expression based on this expression with ternary operation applied to three columns."""
        value = op(
            self._get_ibis_column(col_id_1),
            self._get_ibis_column(col_id_2),
            self._get_ibis_column(col_id_3),
        ).name(output_column_id)
        return self._set_or_replace_by_id(output_column_id, value)

    def aggregate(
        self,
        aggregations: typing.Sequence[typing.Tuple[str, agg_ops.AggregateOp, str]],
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
        table = self._to_ibis_expr("unordered")
        stats = {
            col_out: agg_op._as_ibis(table[col_in])
            for col_in, agg_op, col_out in aggregations
        }
        if by_column_ids:
            result = table.group_by(by_column_ids).aggregate(**stats)
            # Must have deterministic ordering, so order by the unique "by" column
            ordering = ExpressionOrdering(
                [
                    OrderingColumnReference(column_id=column_id)
                    for column_id in by_column_ids
                ],
                total_ordering_columns=frozenset(by_column_ids),
            )
            columns = tuple(result[key] for key in result.columns)
            expr = ArrayValue(self._session, result, columns=columns, ordering=ordering)
            if dropna:
                for column_id in by_column_ids:
                    expr = expr._filter(
                        ops.notnull_op._as_ibis(expr._get_ibis_column(column_id))
                    )
            # Can maybe remove this as Ordering id is redundant as by_column is unique after aggregation
            return expr._project_offsets()
        else:
            aggregates = {**stats, ORDER_ID_COLUMN: ibis_types.literal(0)}
            result = table.aggregate(**aggregates)
            # Ordering is irrelevant for single-row output, but set ordering id regardless as other ops(join etc.) expect it.
            ordering = ExpressionOrdering(
                ordering_value_columns=[OrderingColumnReference(ORDER_ID_COLUMN)],
                total_ordering_columns=frozenset([ORDER_ID_COLUMN]),
                integer_encoding=IntegerEncoding(is_encoded=True, is_sequential=True),
            )
            return ArrayValue(
                self._session,
                result,
                columns=[result[col_id] for col_id in [*stats.keys()]],
                hidden_ordering_columns=[result[ORDER_ID_COLUMN]],
                ordering=ordering,
            )

    def corr_aggregate(
        self, corr_aggregations: typing.Sequence[typing.Tuple[str, str, str]]
    ) -> ArrayValue:
        """
        Get correlations between each lef_column_id and right_column_id, stored in the respective output_column_id.
        This uses BigQuery's CORR under the hood, and thus only Pearson's method is used.
        Arguments:
            corr_aggregations: left_column_id, right_column_id, output_column_id tuples
        """
        table = self._to_ibis_expr("unordered")
        stats = {
            col_out: table[col_left].corr(table[col_right], how="pop")
            for col_left, col_right, col_out in corr_aggregations
        }
        aggregates = {**stats, ORDER_ID_COLUMN: ibis_types.literal(0)}
        result = table.aggregate(**aggregates)
        # Ordering is irrelevant for single-row output, but set ordering id regardless as other ops(join etc.) expect it.
        ordering = ExpressionOrdering(
            ordering_value_columns=[OrderingColumnReference(ORDER_ID_COLUMN)],
            total_ordering_columns=frozenset([ORDER_ID_COLUMN]),
            integer_encoding=IntegerEncoding(is_encoded=True, is_sequential=True),
        )
        return ArrayValue(
            self._session,
            result,
            columns=[result[col_id] for col_id in [*stats.keys()]],
            hidden_ordering_columns=[result[ORDER_ID_COLUMN]],
            ordering=ordering,
        )

    def project_window_op(
        self,
        column_name: str,
        op: agg_ops.WindowOp,
        window_spec: WindowSpec,
        output_name=None,
        *,
        never_skip_nulls=False,
        skip_reproject_unsafe: bool = False,
    ) -> ArrayValue:
        """
        Creates a new expression based on this expression with unary operation applied to one column.
        column_name: the id of the input column present in the expression
        op: the windowable operator to apply to the input column
        window_spec: a specification of the window over which to apply the operator
        output_name: the id to assign to the output of the operator, by default will replace input col if distinct output id not provided
        never_skip_nulls: will disable null skipping for operators that would otherwise do so
        skip_reproject_unsafe: skips the reprojection step, can be used when performing many non-dependent window operations, user responsible for not nesting window expressions, or using outputs as join, filter or aggregation keys before a reprojection
        """
        column = typing.cast(ibis_types.Column, self._get_ibis_column(column_name))
        window = self._ibis_window_from_spec(window_spec, allow_ties=op.handles_ties)

        window_op = op._as_ibis(column, window)

        clauses = []
        if op.skips_nulls and not never_skip_nulls:
            clauses.append((column.isnull(), ibis.NA))
        if window_spec.min_periods:
            if op.skips_nulls:
                # Most operations do not count NULL values towards min_periods
                observation_count = agg_ops.count_op._as_ibis(column, window)
            else:
                # Operations like count treat even NULLs as valid observations for the sake of min_periods
                # notnull is just used to convert null values to non-null (FALSE) values to be counted
                denulled_value = typing.cast(ibis_types.BooleanColumn, column.notnull())
                observation_count = agg_ops.count_op._as_ibis(denulled_value, window)
            clauses.append(
                (
                    observation_count < ibis_types.literal(window_spec.min_periods),
                    ibis.NA,
                )
            )
        if clauses:
            case_statement = ibis.case()
            for clause in clauses:
                case_statement = case_statement.when(clause[0], clause[1])
            case_statement = case_statement.else_(window_op).end()
            window_op = case_statement

        result = self._set_or_replace_by_id(output_name or column_name, window_op)
        # TODO(tbergeron): Automatically track analytic expression usage and defer reprojection until required for valid query generation.
        return result._reproject_to_table() if not skip_reproject_unsafe else result

    def to_sql(
        self,
        offset_column: typing.Optional[str] = None,
        col_id_overrides: typing.Mapping[str, str] = {},
        sorted: bool = False,
    ) -> str:
        offsets_id = offset_column or ORDER_ID_COLUMN

        sql = self._session.ibis_client.compile(
            self._to_ibis_expr(
                ordering_mode="offset_col"
                if (offset_column or sorted)
                else "unordered",
                order_col_name=offsets_id,
                col_id_overrides=col_id_overrides,
            )
        )
        if sorted:
            sql = textwrap.dedent(
                f"""
                SELECT * EXCEPT (`{offsets_id}`)
                FROM ({sql})
                ORDER BY `{offsets_id}`
                """
            )
        return typing.cast(str, sql)

    def _to_ibis_expr(
        self,
        ordering_mode: Literal["string_encoded", "offset_col", "unordered"],
        order_col_name: Optional[str] = ORDER_ID_COLUMN,
        expose_hidden_cols: bool = False,
        fraction: Optional[float] = None,
        col_id_overrides: typing.Mapping[str, str] = {},
    ):
        """
        Creates an Ibis table expression representing the DataFrame.

        ArrayValue objects are sorted, so the following options are available
        to reflect this in the ibis expression.

        * "offset_col": Zero-based offsets are generated as a column, this will
          not sort the rows however.
        * "string_encoded": An ordered string column is provided in output table.
        * "unordered": No ordering information will be provided in output. Only
          value columns are projected.

        For offset or ordered column, order_col_name can be used to assign the
        output label for the ordering column. If none is specified, the default
        column name will be 'bigframes_ordering_id'

        Args:
            ordering_mode:
                How to construct the Ibis expression from the ArrayValue. See
                above for details.
            order_col_name:
                If the ordering mode outputs a single ordering or offsets
                column, use this as the column name.
            expose_hidden_cols:
                If True, include the hidden ordering columns in the results.
                Only compatible with `order_by` and `unordered`
                ``ordering_mode``.
            col_id_overrides:
                overrides the column ids for the result
        Returns:
            An ibis expression representing the data help by the ArrayValue object.
        """
        assert ordering_mode in (
            "string_encoded",
            "offset_col",
            "unordered",
        )
        if expose_hidden_cols and ordering_mode in ("ordered_col", "offset_col"):
            raise ValueError(
                f"Cannot expose hidden ordering columns with ordering_mode {ordering_mode}"
            )

        columns = list(self._columns)
        columns_to_drop: list[
            str
        ] = []  # Ordering/Filtering columns that will be dropped at end

        if self._reduced_predicate is not None:
            columns.append(self._reduced_predicate)
            # Usually drop predicate as it is will be all TRUE after filtering
            if not expose_hidden_cols:
                columns_to_drop.append(self._reduced_predicate.get_name())

        order_columns = self._create_order_columns(
            ordering_mode, order_col_name, expose_hidden_cols
        )
        columns.extend(order_columns)

        # Special case for empty tables, since we can't create an empty
        # projection.
        if not columns:
            return ibis.memtable([])

        # Make sure all dtypes are the "canonical" ones for BigFrames. This is
        # important for operations like UNION where the schema must match.
        table = self._table.select(
            bigframes.dtypes.ibis_value_to_canonical_type(column) for column in columns
        )
        base_table = table
        if self._reduced_predicate is not None:
            table = table.filter(base_table[PREDICATE_COLUMN])
        table = table.drop(*columns_to_drop)
        if col_id_overrides:
            table = table.relabel(col_id_overrides)
        if fraction is not None:
            table = table.filter(ibis.random() < ibis.literal(fraction))
        return table

    def _create_order_columns(
        self,
        ordering_mode: str,
        order_col_name: Optional[str],
        expose_hidden_cols: bool,
    ) -> typing.Sequence[ibis_types.Value]:
        # Generate offsets if current ordering id semantics are not sufficiently strict
        if ordering_mode == "offset_col":
            return (self._create_offset_column().name(order_col_name),)
        elif ordering_mode == "string_encoded":
            return (self._create_string_ordering_column().name(order_col_name),)
        elif expose_hidden_cols:
            return self.hidden_ordering_columns
        return ()

    def _create_offset_column(self) -> ibis_types.IntegerColumn:
        if self._ordering.total_order_col and self._ordering.is_sequential:
            offsets = self._get_any_column(self._ordering.total_order_col.column_id)
            return typing.cast(ibis_types.IntegerColumn, offsets)
        else:
            window = ibis.window(order_by=self._ibis_order)
            if self._predicates:
                window = window.group_by(self._reduced_predicate)
            offsets = ibis.row_number().over(window)
            return typing.cast(ibis_types.IntegerColumn, offsets)

    def _create_string_ordering_column(self) -> ibis_types.StringColumn:
        if self._ordering.total_order_col and self._ordering.is_string_encoded:
            string_order_ids = self._get_any_column(
                self._ordering.total_order_col.column_id
            )
            return typing.cast(ibis_types.StringColumn, string_order_ids)
        if (
            self._ordering.total_order_col
            and self._ordering.integer_encoding.is_encoded
        ):
            # Special case: non-negative integer ordering id can be converted directly to string without regenerating row numbers
            int_values = self._get_any_column(self._ordering.total_order_col.column_id)
            return encode_order_string(
                typing.cast(ibis_types.IntegerColumn, int_values),
            )
        else:
            # Have to build string from scratch
            window = ibis.window(order_by=self._ibis_order)
            if self._predicates:
                window = window.group_by(self._reduced_predicate)
            row_nums = typing.cast(
                ibis_types.IntegerColumn, ibis.row_number().over(window)
            )
            return encode_order_string(row_nums)

    def start_query(
        self,
        job_config: Optional[bigquery.job.QueryJobConfig] = None,
        max_results: Optional[int] = None,
        *,
        sorted: bool = True,
    ) -> Tuple[bigquery.table.RowIterator, bigquery.QueryJob]:
        """Execute a query and return metadata about the results."""
        # TODO(swast): Cache the job ID so we can look it up again if they ask
        # for the results? We'd need a way to invalidate the cache if DataFrame
        # becomes mutable, though. Or move this method to the immutable
        # expression class.
        # TODO(swast): We might want to move this method to Session and/or
        # provide our own minimal metadata class. Tight coupling to the
        # BigQuery client library isn't ideal, especially if we want to support
        # a LocalSession for unit testing.
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        sql = self.to_sql(sorted=True)  # type:ignore
        return self._session._start_query(
            sql=sql,
            job_config=job_config,
            max_results=max_results,
        )

    def _get_table_size(self, destination_table):
        return self._session._get_table_size(destination_table)

    def _reproject_to_table(self) -> ArrayValue:
        """
        Internal operators that projects the internal representation into a
        new ibis table expression where each value column is a direct
        reference to a column in that table expression. Needed after
        some operations such as window operations that cannot be used
        recursively in projections.
        """
        table = self._to_ibis_expr(
            "unordered",
            expose_hidden_cols=True,
        )
        columns = [table[column_name] for column_name in self._column_names]
        ordering_col_ids = [
            ref.column_id for ref in self._ordering.all_ordering_columns
        ]
        hidden_ordering_columns = [
            table[column_name]
            for column_name in self._hidden_ordering_column_names
            if column_name in ordering_col_ids
        ]
        return ArrayValue(
            self._session,
            table,
            columns=columns,
            hidden_ordering_columns=hidden_ordering_columns,
            ordering=self._ordering,
        )

    def _ibis_window_from_spec(self, window_spec: WindowSpec, allow_ties: bool = False):
        group_by: typing.List[ibis_types.Value] = (
            [
                typing.cast(
                    ibis_types.Column, _as_identity(self._get_ibis_column(column))
                )
                for column in window_spec.grouping_keys
            ]
            if window_spec.grouping_keys
            else []
        )
        if self._reduced_predicate is not None:
            group_by.append(self._reduced_predicate)
        if window_spec.ordering:
            order_by = _convert_ordering_to_table_values(
                {**self._column_names, **self._hidden_ordering_column_names},
                window_spec.ordering,
            )
            if not allow_ties:
                # Most operator need an unambiguous ordering, so the table's total ordering is appended
                order_by = tuple([*order_by, *self._ibis_order])
        elif (window_spec.following is not None) or (window_spec.preceding is not None):
            # If window spec has following or preceding bounds, we need to apply an unambiguous ordering.
            order_by = tuple(self._ibis_order)
        else:
            # Unbound grouping window. Suitable for aggregations but not for analytic function application.
            order_by = None
        return ibis.window(
            preceding=window_spec.preceding,
            following=window_spec.following,
            order_by=order_by,
            group_by=group_by,
        )

    def unpivot(
        self,
        row_labels: typing.Sequence[typing.Hashable],
        unpivot_columns: typing.Sequence[
            typing.Tuple[str, typing.Sequence[typing.Optional[str]]]
        ],
        *,
        passthrough_columns: typing.Sequence[str] = (),
        index_col_ids: typing.Sequence[str] = ["index"],
        dtype: typing.Union[
            bigframes.dtypes.Dtype, typing.Sequence[bigframes.dtypes.Dtype]
        ] = pandas.Float64Dtype(),
        how="left",
    ) -> ArrayValue:
        """
        Unpivot ArrayValue columns.

        Args:
            row_labels: Identifies the source of the row. Must be equal to length to source column list in unpivot_columns argument.
            unpivot_columns: Mapping of column id to list of input column ids. Lists of input columns may use None.
            passthrough_columns: Columns that will not be unpivoted. Column id will be preserved.
            index_col_id (str): The column id to be used for the row labels.
            dtype (dtype or list of dtype): Dtype to use for the unpivot columns. If list, must be equal in number to unpivot_columns.

        Returns:
            ArrayValue: The unpivoted ArrayValue
        """
        if how not in ("left", "right"):
            raise ValueError("'how' must be 'left' or 'right'")
        table = self._to_ibis_expr("unordered", expose_hidden_cols=True)
        row_n = len(row_labels)
        hidden_col_ids = self._hidden_ordering_column_names.keys()
        if not all(
            len(source_columns) == row_n for _, source_columns in unpivot_columns
        ):
            raise ValueError("Columns and row labels must all be same length.")

        unpivot_offset_id = bigframes.core.guid.generate_guid("unpivot_offsets_")
        unpivot_table = table.cross_join(
            ibis.memtable({unpivot_offset_id: range(row_n)})
        )
        # Use ibis memtable to infer type of rowlabels (if possible)
        # TODO: Allow caller to specify dtype
        if isinstance(row_labels[0], tuple):
            labels_table = ibis.memtable(row_labels)
            labels_ibis_types = [
                labels_table[col].type() for col in labels_table.columns
            ]
        else:
            labels_ibis_types = [ibis.memtable({"col": row_labels})["col"].type()]
        labels_dtypes = [
            bigframes.dtypes.ibis_dtype_to_bigframes_dtype(ibis_type)
            for ibis_type in labels_ibis_types
        ]

        label_columns = []
        for label_part, (col_id, label_dtype) in enumerate(
            zip(index_col_ids, labels_dtypes)
        ):
            # interpret as tuples even if it wasn't originally so can apply same logic for multi-column labels
            labels_as_tuples = [
                label if isinstance(label, tuple) else (label,) for label in row_labels
            ]
            cases = [
                (
                    i,
                    bigframes.dtypes.literal_to_ibis_scalar(
                        label_tuple[label_part],  # type:ignore
                        force_dtype=label_dtype,  # type:ignore
                    ),
                )
                for i, label_tuple in enumerate(labels_as_tuples)
            ]
            labels_value = (
                typing.cast(ibis_types.IntegerColumn, unpivot_table[unpivot_offset_id])
                .cases(cases, default=None)  # type:ignore
                .name(col_id)
            )
            label_columns.append(labels_value)

        unpivot_values = []
        for j in range(len(unpivot_columns)):
            col_dtype = dtype[j] if utils.is_list_like(dtype) else dtype
            result_col, source_cols = unpivot_columns[j]
            null_value = bigframes.dtypes.literal_to_ibis_scalar(
                None, force_dtype=col_dtype
            )
            ibis_values = [
                ops.AsTypeOp(col_dtype)._as_ibis(unpivot_table[col])
                if col is not None
                else null_value
                for col in source_cols
            ]
            cases = [(i, ibis_values[i]) for i in range(len(ibis_values))]
            unpivot_value = typing.cast(
                ibis_types.IntegerColumn, unpivot_table[unpivot_offset_id]
            ).cases(
                cases, default=null_value  # type:ignore
            )
            unpivot_values.append(unpivot_value.name(result_col))

        unpivot_table = unpivot_table.select(
            passthrough_columns,
            *label_columns,
            *unpivot_values,
            *hidden_col_ids,
            unpivot_offset_id,
        )

        # Extend the original ordering using unpivot_offset_id
        old_ordering = self._ordering
        if how == "left":
            new_ordering = ExpressionOrdering(
                ordering_value_columns=[
                    *old_ordering.ordering_value_columns,
                    OrderingColumnReference(unpivot_offset_id),
                ],
                total_ordering_columns=frozenset(
                    [*old_ordering.total_ordering_columns, unpivot_offset_id]
                ),
            )
        else:  # how=="right"
            new_ordering = ExpressionOrdering(
                ordering_value_columns=[
                    OrderingColumnReference(unpivot_offset_id),
                    *old_ordering.ordering_value_columns,
                ],
                total_ordering_columns=frozenset(
                    [*old_ordering.total_ordering_columns, unpivot_offset_id]
                ),
            )
        value_columns = [
            unpivot_table[value_col_id] for value_col_id, _ in unpivot_columns
        ]
        passthrough_values = [unpivot_table[col] for col in passthrough_columns]
        hidden_ordering_columns = [
            unpivot_table[unpivot_offset_id],
            *[unpivot_table[hidden_col] for hidden_col in hidden_col_ids],
        ]
        return ArrayValue(
            session=self._session,
            table=unpivot_table,
            columns=[
                *[unpivot_table[col_id] for col_id in index_col_ids],
                *value_columns,
                *passthrough_values,
            ],
            hidden_ordering_columns=hidden_ordering_columns,
            ordering=new_ordering,
        )

    def assign(self, source_id: str, destination_id: str) -> ArrayValue:
        return self._set_or_replace_by_id(
            destination_id, self._get_ibis_column(source_id)
        )

    def assign_constant(
        self,
        destination_id: str,
        value: typing.Any,
        dtype: typing.Optional[bigframes.dtypes.Dtype],
    ) -> ArrayValue:
        # TODO(b/281587571): Solve scalar constant aggregation problem w/Ibis.
        ibis_value = bigframes.dtypes.literal_to_ibis_scalar(value, dtype)
        if ibis_value is None:
            raise NotImplementedError(
                f"Type not supported as scalar value {type(value)}. {constants.FEEDBACK_LINK}"
            )
        expr = self._set_or_replace_by_id(destination_id, ibis_value)
        return expr._reproject_to_table()

    def _set_or_replace_by_id(self, id: str, new_value: ibis_types.Value) -> ArrayValue:
        """Safely assign by id while maintaining ordering integrity."""
        # TODO: Split into explicit set and replace methods
        ordering_col_ids = [
            col_ref.column_id for col_ref in self._ordering.ordering_value_columns
        ]
        if id in ordering_col_ids:
            return self._hide_column(id)._set_or_replace_by_id(id, new_value)

        builder = self.builder()
        if id in self.column_ids:
            builder.columns = [
                val if (col_id != id) else new_value.name(id)
                for col_id, val in zip(self.column_ids, self._columns)
            ]
        else:
            builder.columns = [*self.columns, new_value.name(id)]
        return builder.build()

    def cached(self, cluster_cols: typing.Sequence[str]) -> ArrayValue:
        """Write the ArrayValue to a session table and create a new block object that references it."""
        ibis_expr = self._to_ibis_expr("unordered", expose_hidden_cols=True)
        destination = self._session._ibis_to_session_table(
            ibis_expr, cluster_cols=cluster_cols, api_name="cache"
        )
        table_expression = self._session.ibis_client.table(
            f"{destination.project}.{destination.dataset_id}.{destination.table_id}"
        )
        new_columns = [table_expression[column] for column in self.column_ids]
        new_hidden_columns = [
            table_expression[column] for column in self._hidden_ordering_column_names
        ]
        return ArrayValue(
            self._session,
            table_expression,
            columns=new_columns,
            hidden_ordering_columns=new_hidden_columns,
            ordering=self._ordering,
        )


class ArrayValueBuilder:
    """Mutable expression class.
    Use ArrayValue.builder() to create from a ArrayValue object.
    """

    def __init__(
        self,
        session: Session,
        table: ibis_types.Table,
        ordering: ExpressionOrdering,
        columns: Collection[ibis_types.Value] = (),
        hidden_ordering_columns: Collection[ibis_types.Value] = (),
        predicates: Optional[Collection[ibis_types.BooleanValue]] = None,
    ):
        self.session = session
        self.table = table
        self.columns = list(columns)
        self.hidden_ordering_columns = list(hidden_ordering_columns)
        self.ordering = ordering
        self.predicates = list(predicates) if predicates is not None else None

    def build(self) -> ArrayValue:
        return ArrayValue(
            session=self.session,
            table=self.table,
            columns=self.columns,
            hidden_ordering_columns=self.hidden_ordering_columns,
            ordering=self.ordering,
            predicates=self.predicates,
        )


def _reduce_predicate_list(
    predicate_list: typing.Collection[ibis_types.BooleanValue],
) -> ibis_types.BooleanValue:
    """Converts a list of predicates BooleanValues into a single BooleanValue."""
    if len(predicate_list) == 0:
        raise ValueError("Cannot reduce empty list of predicates")
    if len(predicate_list) == 1:
        (item,) = predicate_list
        return item
    return functools.reduce(lambda acc, pred: acc.__and__(pred), predicate_list)


def _convert_ordering_to_table_values(
    value_lookup: typing.Mapping[str, ibis_types.Value],
    ordering_columns: typing.Sequence[OrderingColumnReference],
) -> typing.Sequence[ibis_types.Value]:
    column_refs = ordering_columns
    ordering_values = []
    for ordering_col in column_refs:
        column = typing.cast(ibis_types.Column, value_lookup[ordering_col.column_id])
        ordering_value = (
            ibis.asc(column)
            if ordering_col.direction.is_ascending
            else ibis.desc(column)
        )
        # Bigquery SQL considers NULLS to be "smallest" values, but we need to override in these cases.
        if (not ordering_col.na_last) and (not ordering_col.direction.is_ascending):
            # Force nulls to be first
            is_null_val = typing.cast(ibis_types.Column, column.isnull())
            ordering_values.append(ibis.desc(is_null_val))
        elif (ordering_col.na_last) and (ordering_col.direction.is_ascending):
            # Force nulls to be last
            is_null_val = typing.cast(ibis_types.Column, column.isnull())
            ordering_values.append(ibis.asc(is_null_val))
        ordering_values.append(ordering_value)
    return ordering_values


def _as_identity(value: ibis_types.Value):
    # Some types need to be converted to string to enable groupby
    if value.type().is_float64() or value.type().is_geospatial():
        return value.cast(ibis_dtypes.str)
    return value
