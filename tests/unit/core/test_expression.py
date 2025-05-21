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

import bigframes.core.expression as ex
import bigframes.core.identifiers as ids
import bigframes.dtypes as dtypes
import bigframes.operations as ops


def test_simple_expression_dtype():
    expression = ops.add_op.as_expr("a", "b")
    result = expression.resolve_deferred_types(
        {ids.ColumnId("a"): dtypes.INT_DTYPE, ids.ColumnId("b"): dtypes.INT_DTYPE}
    ).output_type
    assert result == dtypes.INT_DTYPE


def test_nested_expression_dtype():
    expression = ops.add_op.as_expr(
        "a", ops.abs_op.as_expr(ops.sub_op.as_expr("b", ex.const(3.14)))
    )

    result = expression.resolve_deferred_types(
        {ids.ColumnId("a"): dtypes.INT_DTYPE, ids.ColumnId("b"): dtypes.INT_DTYPE}
    ).output_type

    assert result == dtypes.FLOAT_DTYPE


def test_where_op_dtype():
    expression = ops.where_op.as_expr(ex.const(3), ex.const(True), ex.const(None))

    result = expression.resolve_deferred_types({}).output_type

    assert result == dtypes.INT_DTYPE


def test_astype_op_dtype():
    expression = ops.AsTypeOp(dtypes.INT_DTYPE).as_expr(ex.const(3.14159))

    result = expression.resolve_deferred_types({}).output_type

    assert result == dtypes.INT_DTYPE


def test_deref_op_default_dtype_is_deferred():
    expression = ex.deref("mycol")

    assert expression.output_type == dtypes.ABSENT_DTYPE


def test_deref_op_dtype_resolution():
    expression = ex.deref("mycol")

    result = expression.resolve_deferred_types(
        {ids.ColumnId("mycol"): dtypes.STRING_DTYPE}
    ).output_type

    assert result == dtypes.STRING_DTYPE


def test_deref_op_dtype_resolution_short_circuit():
    expression = ex.deref("myCol", dtypes.INT_DTYPE)

    result = expression.resolve_deferred_types(
        {ids.ColumnId("anotherCol"): dtypes.STRING_DTYPE}
    ).output_type

    assert result == dtypes.INT_DTYPE


def test_nested_expression_dtypes_are_cached():
    expression = ops.add_op.as_expr(ex.deref("left_col"), ex.deref("right_col"))

    expression = expression.resolve_deferred_types(
        {
            ids.ColumnId("right_col"): dtypes.INT_DTYPE,
            ids.ColumnId("left_col"): dtypes.FLOAT_DTYPE,
        }
    )

    assert expression.output_type == dtypes.FLOAT_DTYPE
    assert isinstance(expression, ex.OpExpression)
    assert expression.inputs[0].output_type == dtypes.FLOAT_DTYPE
    assert expression.inputs[1].output_type == dtypes.INT_DTYPE
