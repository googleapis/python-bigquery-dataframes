# Copyright 2025 Google LLC
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

import typing

import pytest

from bigframes.core import agg_expressions as agg_exprs
from bigframes.core import (
    array_value,
    expression,
    identifiers,
    nodes,
    ordering,
    window_spec,
)
from bigframes.operations import aggregations as agg_ops
import bigframes.pandas as bpd

pytest.importorskip("pytest_snapshot")


def _apply_unary_agg_ops(
    obj: bpd.DataFrame,
    ops_list: typing.Sequence[agg_exprs.UnaryAggregation],
    new_names: typing.Sequence[str],
) -> str:
    aggs = [(op, identifiers.ColumnId(name)) for op, name in zip(ops_list, new_names)]

    agg_node = nodes.AggregateNode(obj._block.expr.node, aggregations=tuple(aggs))
    result = array_value.ArrayValue(agg_node)

    sql = result.session._executor.to_sql(result, enable_cache=False)
    return sql


def _apply_unary_window_op(
    obj: bpd.DataFrame,
    op: agg_exprs.UnaryAggregation,
    window_spec: window_spec.WindowSpec,
    new_name: str,
) -> str:
    win_node = nodes.WindowOpNode(
        obj._block.expr.node,
        expression=op,
        window_spec=window_spec,
        output_name=identifiers.ColumnId(new_name),
    )
    result = array_value.ArrayValue(win_node).select_columns([new_name])

    sql = result.session._executor.to_sql(result, enable_cache=False)
    return sql


def test_count(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "int64_col"
    bf_df = scalar_types_df[[col_name]]
    agg_expr = agg_ops.CountOp().as_expr(col_name)
    sql = _apply_unary_agg_ops(bf_df, [agg_expr], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_dense_rank(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "int64_col"
    bf_df = scalar_types_df[[col_name]]
    agg_expr = agg_exprs.UnaryAggregation(
        agg_ops.DenseRankOp(), expression.deref(col_name)
    )
    window = window_spec.WindowSpec(ordering=(ordering.ascending_over(col_name),))
    sql = _apply_unary_window_op(bf_df, agg_expr, window, "agg_int64")

    snapshot.assert_match(sql, "out.sql")


def test_max(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "int64_col"
    bf_df = scalar_types_df[[col_name]]
    agg_expr = agg_ops.MaxOp().as_expr(col_name)
    sql = _apply_unary_agg_ops(bf_df, [agg_expr], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_mean(scalar_types_df: bpd.DataFrame, snapshot):
    col_names = ["int64_col", "bool_col", "duration_col"]
    bf_df = scalar_types_df[col_names]
    bf_df["duration_col"] = bpd.to_timedelta(bf_df["duration_col"], unit="us")

    # The `to_timedelta` creates a new mapping for the column id.
    col_names.insert(0, "rowindex")
    name2id = {
        col_name: col_id
        for col_name, col_id in zip(col_names, bf_df._block.expr.column_ids)
    }

    agg_ops_map = {
        "int64_col": agg_ops.MeanOp().as_expr(name2id["int64_col"]),
        "bool_col": agg_ops.MeanOp().as_expr(name2id["bool_col"]),
        "duration_col": agg_ops.MeanOp().as_expr(name2id["duration_col"]),
        "int64_col_w_floor": agg_ops.MeanOp(should_floor_result=True).as_expr(
            name2id["int64_col"]
        ),
    }
    sql = _apply_unary_agg_ops(
        bf_df, list(agg_ops_map.values()), list(agg_ops_map.keys())
    )

    snapshot.assert_match(sql, "out.sql")


def test_median(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df
    ops_map = {
        "int64_col": agg_ops.MedianOp().as_expr("int64_col"),
        "date_col": agg_ops.MedianOp().as_expr("date_col"),
        "string_col": agg_ops.MedianOp().as_expr("string_col"),
    }
    sql = _apply_unary_agg_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))

    snapshot.assert_match(sql, "out.sql")


def test_min(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "int64_col"
    bf_df = scalar_types_df[[col_name]]
    agg_expr = agg_ops.MinOp().as_expr(col_name)
    sql = _apply_unary_agg_ops(bf_df, [agg_expr], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_rank(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "int64_col"
    bf_df = scalar_types_df[[col_name]]
    agg_expr = agg_exprs.UnaryAggregation(agg_ops.RankOp(), expression.deref(col_name))

    window = window_spec.WindowSpec(ordering=(ordering.ascending_over(col_name),))
    sql = _apply_unary_window_op(bf_df, agg_expr, window, "agg_int64")

    snapshot.assert_match(sql, "out.sql")


def test_sum(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df[["int64_col", "bool_col"]]
    agg_ops_map = {
        "int64_col": agg_ops.SumOp().as_expr("int64_col"),
        "bool_col": agg_ops.SumOp().as_expr("bool_col"),
    }
    sql = _apply_unary_agg_ops(
        bf_df, list(agg_ops_map.values()), list(agg_ops_map.keys())
    )

    snapshot.assert_match(sql, "out.sql")
