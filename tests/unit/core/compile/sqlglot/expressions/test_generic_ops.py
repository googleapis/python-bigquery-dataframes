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

import pytest

from bigframes import dtypes
from bigframes import operations as ops
from bigframes.core import expression as ex
import bigframes.pandas as bpd
from bigframes.testing import utils

pytest.importorskip("pytest_snapshot")


def test_astype_int(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df
    to_type = dtypes.INT_DTYPE

    ops_map = {
        "datetime_col": ops.AsTypeOp(to_type=to_type).as_expr("datetime_col"),
        "datetime_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr(
            "datetime_col"
        ),
        "time_col": ops.AsTypeOp(to_type=to_type).as_expr("time_col"),
        "time_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr("time_col"),
        "timestamp_col": ops.AsTypeOp(to_type=to_type).as_expr("timestamp_col"),
        "numeric_col": ops.AsTypeOp(to_type=to_type).as_expr("numeric_col"),
        "float64_col": ops.AsTypeOp(to_type=to_type).as_expr("float64_col"),
        "float64_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr(
            "float64_col"
        ),
        "str_const": ops.AsTypeOp(to_type=to_type).as_expr(ex.const("100")),
    }

    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_float(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df
    to_type = dtypes.FLOAT_DTYPE

    ops_map = {
        "bool_col": ops.AsTypeOp(to_type=to_type).as_expr("bool_col"),
        "str_const": ops.AsTypeOp(to_type=to_type).as_expr(ex.const("1.34235e4")),
        "bool_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr("bool_col"),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_bool(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df
    to_type = dtypes.BOOL_DTYPE

    ops_map = {
        "bool_col": ops.AsTypeOp(to_type=to_type).as_expr("bool_col"),
        "float64_col": ops.AsTypeOp(to_type=to_type).as_expr("float64_col"),
        "float64_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr(
            "float64_col"
        ),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_time_like(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df

    ops_map = {
        "int64_to_datetime": ops.AsTypeOp(to_type=dtypes.DATETIME_DTYPE).as_expr(
            "int64_col"
        ),
        "int64_to_time": ops.AsTypeOp(to_type=dtypes.TIME_DTYPE).as_expr("int64_col"),
        "int64_to_timestamp": ops.AsTypeOp(to_type=dtypes.TIMESTAMP_DTYPE).as_expr(
            "int64_col"
        ),
        "int64_to_time_safe": ops.AsTypeOp(
            to_type=dtypes.TIME_DTYPE, safe=True
        ).as_expr("int64_col"),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_string(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df
    to_type = dtypes.STRING_DTYPE

    ops_map = {
        "int64_col": ops.AsTypeOp(to_type=to_type).as_expr("int64_col"),
        "bool_col": ops.AsTypeOp(to_type=to_type).as_expr("bool_col"),
        "bool_w_safe": ops.AsTypeOp(to_type=to_type, safe=True).as_expr("bool_col"),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_json(scalar_types_df: bpd.DataFrame, snapshot):
    bf_df = scalar_types_df

    ops_map = {
        "int64_col": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE).as_expr("int64_col"),
        "float64_col": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE).as_expr("float64_col"),
        "bool_col": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE).as_expr("bool_col"),
        "string_col": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE).as_expr("string_col"),
        "bool_w_safe": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE, safe=True).as_expr(
            "bool_col"
        ),
        "string_w_safe": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE, safe=True).as_expr(
            "string_col"
        ),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_from_json(json_types_df: bpd.DataFrame, snapshot):
    bf_df = json_types_df

    ops_map = {
        "int64_col": ops.AsTypeOp(to_type=dtypes.INT_DTYPE).as_expr("json_col"),
        "float64_col": ops.AsTypeOp(to_type=dtypes.FLOAT_DTYPE).as_expr("json_col"),
        "bool_col": ops.AsTypeOp(to_type=dtypes.BOOL_DTYPE).as_expr("json_col"),
        "string_col": ops.AsTypeOp(to_type=dtypes.STRING_DTYPE).as_expr("json_col"),
        "int64_w_safe": ops.AsTypeOp(to_type=dtypes.INT_DTYPE, safe=True).as_expr(
            "json_col"
        ),
    }
    sql = utils._apply_unary_ops(bf_df, list(ops_map.values()), list(ops_map.keys()))
    snapshot.assert_match(sql, "out.sql")


def test_astype_json_invalid(
    scalar_types_df: bpd.DataFrame, json_types_df: bpd.DataFrame
):
    # Test invalid cast to JSON
    with pytest.raises(TypeError, match="Cannot cast timestamp.* to .*json.*"):
        ops_map_to = {
            "datetime_to_json": ops.AsTypeOp(to_type=dtypes.JSON_DTYPE).as_expr(
                "datetime_col"
            ),
        }
        utils._apply_unary_ops(
            scalar_types_df, list(ops_map_to.values()), list(ops_map_to.keys())
        )

    # Test invalid cast from JSON
    with pytest.raises(TypeError, match="Cannot cast .*json.* to timestamp.*"):
        ops_map_from = {
            "json_to_datetime": ops.AsTypeOp(to_type=dtypes.DATETIME_DTYPE).as_expr(
                "json_col"
            ),
        }
        utils._apply_unary_ops(
            json_types_df, list(ops_map_from.values()), list(ops_map_from.keys())
        )


def test_hash(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "string_col"
    bf_df = scalar_types_df[[col_name]]
    sql = utils._apply_unary_ops(bf_df, [ops.hash_op.as_expr(col_name)], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_isnull(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "float64_col"
    bf_df = scalar_types_df[[col_name]]
    sql = utils._apply_unary_ops(bf_df, [ops.isnull_op.as_expr(col_name)], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_notnull(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "float64_col"
    bf_df = scalar_types_df[[col_name]]
    sql = utils._apply_unary_ops(bf_df, [ops.notnull_op.as_expr(col_name)], [col_name])

    snapshot.assert_match(sql, "out.sql")


def test_map(scalar_types_df: bpd.DataFrame, snapshot):
    col_name = "string_col"
    bf_df = scalar_types_df[[col_name]]
    sql = utils._apply_unary_ops(
        bf_df,
        [ops.MapOp(mappings=(("value1", "mapped1"),)).as_expr(col_name)],
        [col_name],
    )

    snapshot.assert_match(sql, "out.sql")
