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

"""SQL escape hatch features."""

from __future__ import annotations

from typing import Sequence

import google.cloud.bigquery

import bigframes.core.expression as expression
import bigframes.core.sql
import bigframes.dataframe
import bigframes.dtypes
import bigframes.operations
import bigframes.series


def sql_scalar(
    sql_template: str,
    columns: Sequence[bigframes.series.Series],
) -> bigframes.series.Series:
    """Create a Series from a SQL template.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.bigquery as bbq
        >>> bpd.options.display.progress_bar = None

        >>> s = bpd.Series([1.5, 2.5, 3.5])
        >>> bbq.sql_scalar("ROUND({0}, 0, 'ROUND_HALF_EVEN')", [s])
        0    2.0
        1    2.0
        2    4.0
        dtype: Float64

    Args:
        sql_template (str):
            A SQL format string with Python-style {0} placeholders for each of
            the Series objects in ``columns``.
        columns (Sequence[bigframes.pandas.Series]):
            Series objects representing the column inputs to the
            ``sql_template``. Must contain at least one Series.

    Returns:
        bigframes.pandas.Series:
            A Series with the SQL applied.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if len(columns) == 0:
        raise ValueError("Must provide at least one column in columns")

    # To integrate this into our expression trees, we need to get the output
    # type, so we do some manual compilation and a dry run query to get that.
    # Another benefit of this is that if there is a syntax error in the SQL
    # template, then this will fail with an error earlier in the process,
    # aiding users in debugging.
    base_series = columns[0]
    executor = base_series._session._executor
    value_refs, base_block = base_series._align_n(columns)
    values_or_column_ids = []
    for expr in value_refs:
        if isinstance(expr, expression.DerefOp):
            values_or_column_ids.append(expr.id.sql)
        else:
            # Constant value, so make sure we escape it properly for embedding
            # in SQL.
            values_or_column_ids.append(bigframes.core.sql.simple_literal(expr.value))  # type: ignore

    # Use the executor directly, because we want the original column IDs, not
    # the user-friendly column names that block.to_sql_query() would produce.
    base_sql = executor.to_sql(base_block.expr)
    select_sql = sql_template.format(*values_or_column_ids)
    dry_run_sql = f"SELECT {select_sql} FROM ({base_sql})"
    bqclient = base_series._session.bqclient
    job = bqclient.query(
        dry_run_sql, job_config=google.cloud.bigquery.QueryJobConfig(dry_run=True)
    )

    _, output_type = bigframes.dtypes.convert_schema_field(job.schema[0])
    op = bigframes.operations.SqlScalarOp(
        _output_type=output_type, sql_template=sql_template
    )
    result_block, result_id = base_block.project_expr(op.as_expr(*value_refs))
    return bigframes.series.Series(result_block.select_column(result_id))
