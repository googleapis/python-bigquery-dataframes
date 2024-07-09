# Contains code from https://github.com/ibis-project/ibis/blob/master/ibis/backends/bigquery/registry.py
"""Module to convert from Ibis expression to SQL string."""

import bigframes_vendored.ibis.expr.operations as vendored_ibis_ops


def _safe_cast_to_datetime(translator, op: vendored_ibis_ops.SafeCastToDatetime):
    arg = translator.translate(op.arg)
    return f"SAFE_CAST({arg} AS DATETIME)"


def _array_aggregate(translator, op: vendored_ibis_ops.ArrayAggregate):
    """This method provides the same functionality as the collect() method in Ibis, with
    the added capability of ordering the results using order_by.
    https://github.com/ibis-project/ibis/issues/9170
    """
    arg = translator.translate(op.arg)

    order_by_sql = ""
    if len(op.order_by) > 0:
        order_by = ", ".join([translator.translate(column) for column in op.order_by])
        order_by_sql = f"ORDER BY {order_by}"

    return f"ARRAY_AGG({arg} IGNORE NULLS {order_by_sql})"


patched_ops = {
    vendored_ibis_ops.SafeCastToDatetime: _safe_cast_to_datetime,  # type:ignore
    vendored_ibis_ops.ArrayAggregate: _array_aggregate,  # type:ignore
}
