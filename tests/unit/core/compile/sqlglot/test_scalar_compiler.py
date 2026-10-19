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

import unittest.mock as mock

import bigframes_vendored.sqlglot.expressions as sge
import pytest

import bigframes.core.compile.sqlglot.expression_compiler as expression_compiler
import bigframes.operations as ops
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr


def test_register_unary_op():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeUnaryOp(ops.UnaryOp):
        name = "fake_unary_op"

    fake_op = FakeUnaryOp()
    mock_impl = mock.Mock()

    @compiler.register_unary_op(fake_op)
    def _(expr: TypedExpr) -> sge.Expression:
        mock_impl(expr)
        return sge.Identifier(this="output")

    arg = TypedExpr(sge.Identifier(this="input"), "string")

    result = compiler.compile_row_op(fake_op, [arg])

    assert result == sge.Identifier(this="output")
    mock_impl.assert_called_once_with(arg)


def test_register_unary_op_pass_op():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeUnaryOp(ops.UnaryOp):
        name = "fake_unary_op"

    fake_op = FakeUnaryOp()
    mock_impl = mock.Mock()

    @compiler.register_unary_op(fake_op, pass_op=True)
    def _(expr: TypedExpr, op: ops.UnaryOp) -> sge.Expression:
        mock_impl(expr, op)
        return sge.Identifier(this="output")

    arg = TypedExpr(sge.Identifier(this="input"), "string")

    result = compiler.compile_row_op(fake_op, [arg])

    assert result == sge.Identifier(this="output")
    mock_impl.assert_called_once_with(arg, fake_op)


def test_register_binary_op():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeBinaryOp(ops.BinaryOp):
        name = "fake_binary_op"

    fake_op = FakeBinaryOp()
    mock_impl = mock.Mock()

    @compiler.register_binary_op(fake_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        mock_impl(left, right)
        return sge.Identifier(this="output")

    arg1 = TypedExpr(sge.Identifier(this="input1"), "string")
    arg2 = TypedExpr(sge.Identifier(this="input2"), "string")

    result = compiler.compile_row_op(fake_op, [arg1, arg2])

    assert result == sge.Identifier(this="output")
    mock_impl.assert_called_once_with(arg1, arg2)


def test_register_binary_op_pass_on():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeBinaryOp(ops.BinaryOp):
        name = "fake_binary_op"

    fake_op = FakeBinaryOp()
    mock_impl = mock.Mock()

    @compiler.register_binary_op(fake_op, pass_op=True)
    def _(left: TypedExpr, right: TypedExpr, op: ops.BinaryOp) -> sge.Expression:
        mock_impl(left, right, op)
        return sge.Identifier(this="output")

    arg1 = TypedExpr(sge.Identifier(this="input1"), "string")
    arg2 = TypedExpr(sge.Identifier(this="input2"), "string")
    result = compiler.compile_row_op(fake_op, [arg1, arg2])

    assert result == sge.Identifier(this="output")

    mock_impl.assert_called_once_with(arg1, arg2, fake_op)


def test_register_ternary_op():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeTernaryOp(ops.TernaryOp):
        name = "fake_ternary_op"

    fake_op = FakeTernaryOp()
    mock_impl = mock.Mock()

    @compiler.register_ternary_op(fake_op)
    def _(arg1: TypedExpr, arg2: TypedExpr, arg3: TypedExpr) -> sge.Expression:
        mock_impl(arg1, arg2, arg3)
        return sge.Identifier(this="output")

    arg1 = TypedExpr(sge.Identifier(this="input1"), "string")
    arg2 = TypedExpr(sge.Identifier(this="input2"), "string")
    arg3 = TypedExpr(sge.Identifier(this="input3"), "string")
    result = compiler.compile_row_op(fake_op, [arg1, arg2, arg3])

    assert result == sge.Identifier(this="output")

    mock_impl.assert_called_once_with(arg1, arg2, arg3)


def test_register_nary_op():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeNaryOp(ops.NaryOp):
        name = "fake_nary_op"

    fake_op = FakeNaryOp()
    mock_impl = mock.Mock()

    @compiler.register_nary_op(fake_op)
    def _(*args: TypedExpr) -> sge.Expression:
        mock_impl(*args)
        return sge.Identifier(this="output")

    arg1 = TypedExpr(sge.Identifier(this="input1"), "string")
    arg2 = TypedExpr(sge.Identifier(this="input2"), "string")
    result = compiler.compile_row_op(fake_op, [arg1, arg2])

    assert result == sge.Identifier(this="output")

    mock_impl.assert_called_once_with(arg1, arg2)


def test_register_nary_op_pass_on():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeNaryOp(ops.NaryOp):
        name = "fake_nary_op"

    fake_op = FakeNaryOp()
    mock_impl = mock.Mock()

    @compiler.register_nary_op(fake_op, pass_op=True)
    def _(*args: TypedExpr, op: ops.NaryOp) -> sge.Expression:
        mock_impl(*args, op=op)
        return sge.Identifier(this="output")

    arg1 = TypedExpr(sge.Identifier(this="input1"), "string")
    arg2 = TypedExpr(sge.Identifier(this="input2"), "string")
    arg3 = TypedExpr(sge.Identifier(this="input3"), "string")
    arg4 = TypedExpr(sge.Identifier(this="input4"), "string")

    result = compiler.compile_row_op(fake_op, [arg1, arg2, arg3, arg4])

    assert result == sge.Identifier(this="output")
    mock_impl.assert_called_once_with(arg1, arg2, arg3, arg4, op=fake_op)


def test_unary_op_parentheses():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeEqOp(ops.BinaryOp):
        name = "fake_eq_op"

    class FakeIsTrueOp(ops.UnaryOp):
        name = "fake_is_true_op"

    eq_op = FakeEqOp()
    is_true_op = FakeIsTrueOp()

    @compiler.register_binary_op(eq_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        return sge.EQ(this=left.expr, expression=right.expr)

    @compiler.register_unary_op(is_true_op)
    def _(expr: TypedExpr) -> sge.Expression:
        return sge.EQ(this=expr.expr, expression=sge.convert(True))

    a = TypedExpr(sge.Identifier(this="a"), "int")
    b = TypedExpr(sge.Identifier(this="b"), "int")

    eq_expr = compiler.compile_row_op(eq_op, [a, b])
    eq_typed_expr = TypedExpr(eq_expr, "bool")

    result = compiler.compile_row_op(is_true_op, [eq_typed_expr])

    assert result.sql() == "(a = b) = TRUE"


def test_binary_op_parentheses_left():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeAddOp(ops.BinaryOp):
        name = "fake_add_op"

    class FakeMulOp(ops.BinaryOp):
        name = "fake_mul_op"

    add_op = FakeAddOp()
    mul_op = FakeMulOp()

    @compiler.register_binary_op(add_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        return sge.Add(this=left.expr, expression=right.expr)

    @compiler.register_binary_op(mul_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        return sge.Mul(this=left.expr, expression=right.expr)

    a = TypedExpr(sge.Identifier(this="a"), "int")
    b = TypedExpr(sge.Identifier(this="b"), "int")
    c = TypedExpr(sge.Identifier(this="c"), "int")

    add_expr = compiler.compile_row_op(add_op, [a, b])
    add_typed_expr = TypedExpr(add_expr, "int")

    result = compiler.compile_row_op(mul_op, [add_typed_expr, c])

    assert result.sql() == "(a + b) * c"


def test_binary_op_parentheses_right():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeAddOp(ops.BinaryOp):
        name = "fake_add_op"

    class FakeMulOp(ops.BinaryOp):
        name = "fake_mul_op"

    add_op = FakeAddOp()
    mul_op = FakeMulOp()

    @compiler.register_binary_op(add_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        return sge.Add(this=left.expr, expression=right.expr)

    @compiler.register_binary_op(mul_op)
    def _(left: TypedExpr, right: TypedExpr) -> sge.Expression:
        return sge.Mul(this=left.expr, expression=right.expr)

    a = TypedExpr(sge.Identifier(this="a"), "int")
    b = TypedExpr(sge.Identifier(this="b"), "int")
    c = TypedExpr(sge.Identifier(this="c"), "int")

    add_expr = compiler.compile_row_op(add_op, [b, c])
    add_typed_expr = TypedExpr(add_expr, "int")

    result = compiler.compile_row_op(mul_op, [a, add_typed_expr])

    assert result.sql() == "a * (b + c)"


def test_ternary_op_parentheses():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeBetweenOp(ops.TernaryOp):
        name = "fake_between_op"

    between_op = FakeBetweenOp()

    @compiler.register_ternary_op(between_op)
    def _(arg1: TypedExpr, arg2: TypedExpr, arg3: TypedExpr) -> sge.Expression:
        return sge.Between(this=arg1.expr, low=arg2.expr, high=arg3.expr)

    a = TypedExpr(
        sge.Add(this=sge.Identifier(this="a"), expression=sge.Identifier(this="b")),
        "int",
    )
    low = TypedExpr(
        sge.Add(this=sge.Identifier(this="c"), expression=sge.Identifier(this="d")),
        "int",
    )
    high = TypedExpr(
        sge.Add(this=sge.Identifier(this="e"), expression=sge.Identifier(this="f")),
        "int",
    )

    result = compiler.compile_row_op(between_op, [a, low, high])

    assert result.sql() == "(a + b) BETWEEN (c + d) AND (e + f)"


def test_nary_op_parentheses():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeAllOp(ops.NaryOp):
        name = "fake_all_op"

    all_op = FakeAllOp()

    @compiler.register_nary_op(all_op)
    def _(*args: TypedExpr) -> sge.Expression:
        return sge.And(this=args[0].expr, expression=args[1].expr)

    arg1 = TypedExpr(
        sge.Or(this=sge.Identifier(this="a"), expression=sge.Identifier(this="b")),
        "bool",
    )
    arg2 = TypedExpr(
        sge.Or(this=sge.Identifier(this="c"), expression=sge.Identifier(this="d")),
        "bool",
    )

    result = compiler.compile_row_op(all_op, [arg1, arg2])

    assert result.sql() == "(a OR b) AND (c OR d)"


def test_register_duplicate_op_raises():
    compiler = expression_compiler.ExpressionCompiler()

    class FakeUnaryOp(ops.UnaryOp):
        name = "fake_unary_op"

    fake_op = FakeUnaryOp()

    @compiler.register_unary_op(fake_op)
    def _(expr: TypedExpr) -> sge.Expression:
        return sge.Identifier(this="output")

    with pytest.raises(ValueError):

        @compiler.register_unary_op(fake_op)
        def _(expr: TypedExpr) -> sge.Expression:
            return sge.Identifier(this="output2")
