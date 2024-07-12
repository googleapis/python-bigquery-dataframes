# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/backends/bigquery/compiler.py
"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import ibis.backends.bigquery.compiler as bq_compiler
import ibis.backends.sql.compiler as sql_compiler
import ibis.backends.sql.datatypes as sql_datatypes
import ibis.expr.operations.reductions as ibis_reductions
import sqlglot as sg
import sqlglot.expressions as sge


class BigQueryCompiler(bq_compiler.BigQueryCompiler):
    UNSUPPORTED_OPS = bq_compiler.BigQueryCompiler.UNSUPPORTED_OPS = tuple(
        op
        for op in bq_compiler.BigQueryCompiler.UNSUPPORTED_OPS
        if op != ibis_reductions.Quantile
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
                        # TODO: Data types and names from schema.
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
                        sge.Tuple(expressions=tuple(map(sge.convert, row)))
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

    def visit_FirstNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.FirstValue(this=arg))

    def visit_LastNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.LastValue(this=arg))

    def visit_ToJsonString(self, op, *, arg):
        return self.f.to_json_string(arg)


# Override implementation.
# We monkeypatch individual methods because the class might have already been imported in other modules.
bq_compiler.BigQueryCompiler.visit_InMemoryTable = BigQueryCompiler.visit_InMemoryTable
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
