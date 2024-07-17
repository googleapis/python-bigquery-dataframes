# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/backends/bigquery/compiler.py
"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import ibis.backends.bigquery.compiler as bq_compiler
import ibis.backends.sql.compiler as sql_compiler
import ibis.backends.sql.datatypes as sql_datatypes
import ibis.common.exceptions as com
from ibis.common.temporal import IntervalUnit
import ibis.expr.operations.reductions as ibis_reductions
import sqlglot as sg
import sqlglot.expressions as sge


class BigQueryCompiler(bq_compiler.BigQueryCompiler):
    UNSUPPORTED_OPS = (
        tuple(
            op
            for op in bq_compiler.BigQueryCompiler.UNSUPPORTED_OPS
            if op != ibis_reductions.Quantile
        )
        if hasattr(bq_compiler.BigQueryCompiler, "UNSUPPORTED_OPS")
        else ()
    )

    def visit_InMemoryTable(self, op, *, name, schema, data):
        # Avoid creating temp tables for small data, which is how memtable is
        # used in BigQuery DataFrames. Inspired by:
        # https://github.com/ibis-project/ibis/blob/efa6fb72bf4c790450d00a926d7bd809dade5902/ibis/backends/druid/compiler.py#L95
        tuples = data.to_frame().itertuples(index=False)
        quoted = self.quoted
        columns = [sg.column(col, quoted=quoted) for col in schema.names]
        expr = sge.Unnest(
            expressions=[
                sge.DataType(
                    this=sge.DataType.Type.ARRAY,
                    expressions=[
                        sge.DataType(
                            this=sge.DataType.Type.STRUCT,
                            expressions=[
                                sge.ColumnDef(
                                    this=sge.to_identifier(field, quoted=self.quoted),
                                    kind=sql_datatypes.SqlglotType.from_ibis(type_),
                                )
                                for field, type_ in zip(schema.names, schema.types)
                            ],
                            nested=True,
                        )
                    ],
                    nested=True,
                    values=[
                        sge.Tuple(
                            expressions=tuple(
                                self.visit_Literal(None, value=value, dtype=type_)
                                for value, type_ in zip(row, schema.types)
                            )
                        )
                        for row in tuples
                    ],
                ),
            ],
            alias=sge.TableAlias(
                this=sg.to_identifier(name, quoted=quoted),
                columns=columns,
            ),
        )
        # return expr
        return sg.select(sge.Star()).from_(expr)

    def visit_NonNullLiteral(self, op, *, value, dtype):
        # Patch from https://github.com/ibis-project/ibis/pull/9610 to support ibis 9.0.0 and 9.1.0
        if dtype.is_inet() or dtype.is_macaddr():
            return sge.convert(str(value))
        elif dtype.is_timestamp():
            funcname = "DATETIME" if dtype.timezone is None else "TIMESTAMP"
            return self.f.anon[funcname](value.isoformat())
        elif dtype.is_date():
            return self.f.date_from_parts(value.year, value.month, value.day)
        elif dtype.is_time():
            time = self.f.time_from_parts(value.hour, value.minute, value.second)
            if micros := value.microsecond:
                # bigquery doesn't support `time(12, 34, 56.789101)`, AKA a
                # float seconds specifier, so add any non-zero micros to the
                # time value
                return sge.TimeAdd(
                    this=time, expression=sge.convert(micros), unit=self.v.MICROSECOND
                )
            return time
        elif dtype.is_binary():
            return sge.Cast(
                this=sge.convert(value.hex()),
                to=sge.DataType(this=sge.DataType.Type.BINARY),
                format=sge.convert("HEX"),
            )
        elif dtype.is_interval():
            if dtype.unit == IntervalUnit.NANOSECOND:
                raise com.UnsupportedOperationError(
                    "BigQuery does not support nanosecond intervals"
                )
        elif dtype.is_uuid():
            return sge.convert(str(value))
        return None

    # Custom operators.

    def visit_ArrayAggregate(self, op, *, arg, order_by, where):
        if len(order_by) > 0:
            expr = sge.Order(
                this=arg,
                expressions=[
                    # Avoid adding NULLS FIRST / NULLS LAST in SQL, which is
                    # unsupported in ARRAY_AGG by reconstructing the node as
                    # plain SQL text.
                    f"({order_column.args['this'].sql(dialect='bigquery')}) {'DESC' if order_column.args.get('desc') else 'ASC'}"
                    for order_column in order_by
                ],
            )
        else:
            expr = arg
        return sge.IgnoreNulls(this=self.agg.array_agg(expr, where=where))

    def visit_FirstNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.FirstValue(this=arg))

    def visit_LastNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.LastValue(this=arg))

    def visit_ToJsonString(self, op, *, arg):
        return self.f.to_json_string(arg)


# Override implementation.
# We monkeypatch individual methods because the class might have already been imported in other modules.
bq_compiler.BigQueryCompiler.visit_InMemoryTable = BigQueryCompiler.visit_InMemoryTable
bq_compiler.BigQueryCompiler.visit_NonNullLiteral = (
    BigQueryCompiler.visit_NonNullLiteral
)

# Custom operators.
bq_compiler.BigQueryCompiler.visit_ArrayAggregate = (
    BigQueryCompiler.visit_ArrayAggregate
)
bq_compiler.BigQueryCompiler.visit_FirstNonNullValue = (
    BigQueryCompiler.visit_FirstNonNullValue
)
bq_compiler.BigQueryCompiler.visit_LastNonNullValue = (
    BigQueryCompiler.visit_LastNonNullValue
)
bq_compiler.BigQueryCompiler.visit_ToJsonString = BigQueryCompiler.visit_ToJsonString

# TODO(swast): sqlglot base implementation appears to work fine for the bigquery backend, at least in our windowed contexts. See: ISSUE NUMBER
bq_compiler.BigQueryCompiler.UNSUPPORTED_OPS = BigQueryCompiler.UNSUPPORTED_OPS
bq_compiler.BigQueryCompiler.visit_Quantile = (
    sql_compiler.SQLGlotCompiler.visit_Quantile
)
