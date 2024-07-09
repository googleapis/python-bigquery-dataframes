# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/backends/bigquery/compiler.py
"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import ibis.backends.bigquery.compiler as bq_compiler
import sqlglot as sg
import sqlglot.expressions as sge


class BigQueryCompiler(bq_compiler.BigQueryCompiler):
    def visit_InMemoryTable(self, op, *, name, schema, data):
        # Avoid creating temp tables for small data, which is how memtable is
        # used in BigQuery DataFrames. Implementation from:
        # https://github.com/ibis-project/ibis/blob/efa6fb72bf4c790450d00a926d7bd809dade5902/ibis/backends/druid/compiler.py#L95
        tuples = data.to_frame().itertuples(index=False)
        quoted = self.quoted
        columns = [sg.column(col, quoted=quoted) for col in schema.names]
        expr = sge.Values(
            expressions=[
                sge.Tuple(expressions=tuple(map(sge.convert, row))) for row in tuples
            ],
            alias=sge.TableAlias(
                this=sg.to_identifier(name, quoted=quoted),
                columns=columns,
            ),
        )
        return sg.select(*columns).from_(expr)

    def visit_FirstNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.FirstValue(this=arg))

    def visit_LastNonNullValue(self, op, *, arg):
        return sge.IgnoreNulls(this=sge.LastValue(this=arg))


# Override implementation.
# We monkeypatch individual methods because the class might have already been imported in other modules.
bq_compiler.BigQueryCompiler.visit_InMemoryTable = BigQueryCompiler.visit_InMemoryTable
bq_compiler.BigQueryCompiler.visit_FirstNonNullValue = (
    BigQueryCompiler.visit_FirstNonNullValue
)
bq_compiler.BigQueryCompiler.visit_LastNonNullValue = (
    BigQueryCompiler.visit_LastNonNullValue
)
