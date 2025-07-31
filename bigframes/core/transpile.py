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

from __future__ import annotations

import ast
import functools
import inspect
import traceback
from typing import Callable, Optional

import numpy

from bigframes import operations as ops
from bigframes.core import expression, flow_graph
import bigframes.operations.imperative_ops as imperative_ops
import bigframes.operations.numpy_op_maps as numpy_ops

_OP_MAPPING: dict[type[ast.operator], ops.BinaryOp] = {
    ast.Add: ops.add_op,
    ast.Sub: ops.sub_op,
    ast.Mult: ops.mul_op,
    ast.Div: ops.div_op,
}


def fn_to_expr(func: Callable) -> Optional[expression.Expression]:
    """
    Attempts two different approaches for converting a python function to a bigframes expression.

    1. Accesses the source code and parses it into an ast, which is then converted to bigframes expression.
    2. Fetches the bytecode from the function and converts this into a bigframes expression.

    The first approach relies on having a simple top-level function def, but is easier to process.
    Bytecode will always be present, but is hard to convert, and less stable between python versions.
    """
    if not inspect.isfunction(func):
        raise TypeError("Expected a function, not a callable object.")
    # from_ast = transpile_w_ast(func)
    # if from_ast is not None:
    #    return from_ast

    # hail mary, translate bytecode instead. this is needed for lambdas, where ast isn't available
    from_bytecode = dis_to_expr(func)
    if from_bytecode is not None:
        return from_bytecode

    return None


def dis_to_expr(func: Callable) -> Optional[expression.Expression]:
    try:
        transpiler = flow_graph.SSATranspiler(func)
        print("--- SSA CFG ---")
        print(transpiler.cfg.to_dot())
        return transpiler.to_sql_expr()
    except Exception:
        print(f"Error transpiling bytecode: {traceback.format_exc()}")
        return None


def transpile_w_ast(func: Callable) -> Optional[expression.Expression]:
    func_globals = func.__globals__
    numpy_module_names = set(
        key for key, value in func_globals.items() if value == numpy
    )

    @functools.singledispatch
    def ast_to_expr(root: ast.AST) -> expression.Expression:
        """ "
        Essentially a very simple python ast -> bigframes expression transpiler.

        Recognizes simple expressions (arithmetic, ternary ops).
        Will fail on loading almost anything other than a constant.
        """
        raise NotImplementedError(f"Cannot convert {root} to scalar expression.")

    @ast_to_expr.register
    def _(root: ast.FunctionDef) -> expression.Expression:
        for stmt in root.body:
            if not isinstance(stmt, ast.Return):
                raise NotImplementedError("Can only return statements")
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return expression.const(None)
                return ast_to_expr(stmt.value)
        return expression.const(None)

    @ast_to_expr.register
    def _(root: ast.Assign) -> expression.Expression:
        if len(root.targets) != 1:
            raise NotImplementedError("Cannot handle multiple assignment targets")
        target = root.targets[0]
        if not isinstance(target, ast.Name):
            raise NotImplementedError("Can only handle assigning to names")
        return imperative_ops.AssignmentOp(variable=target.id).as_expr(
            ast_to_expr(root.value)
        )

    @ast_to_expr.register
    def _(root: ast.Expr):
        return ast_to_expr(root.value)

    @ast_to_expr.register
    def _(root: ast.BinOp):
        if type(root.op) in _OP_MAPPING:
            left_expr = ast_to_expr(root.left)
            right_expr = ast_to_expr(root.right)
            return _OP_MAPPING[type(root.op)].as_expr(left_expr, right_expr)
        raise NotImplementedError(f"Unrecognized op: {root.op}")

    @ast_to_expr.register
    def _(root: ast.Call):
        # Basically only recognizes some numpy functions, and only when called with positional args, not kwargs
        if (
            isinstance(root.func, ast.Attribute)
            and isinstance(root.func.value, ast.Name)
            and (root.func.value.id in numpy_module_names)
        ):
            numpy_func = getattr(numpy, root.func.attr)
            if len(root.args) == 1:
                op = numpy_ops.NUMPY_TO_OP.get(numpy_func, None)  # type: ignore
                if op is None:
                    raise NotImplementedError(f"Unrecognized numpy op: {root.func}")
                return op.as_expr(ast_to_expr(root.args[0]))
            elif len(root.args) == 2:
                binop = numpy_ops.NUMPY_TO_BINOP.get(numpy_func, None)  # type: ignore
                if binop is None:
                    raise NotImplementedError(f"Unrecognized numpy op: {root.func}")
                return binop.as_expr(
                    ast_to_expr(root.args[0]), ast_to_expr(root.args[1])
                )
        raise NotImplementedError(f"Unrecognized op: {root.func}")

    @ast_to_expr.register
    def _(root: ast.Constant):
        return expression.const(root.value)

    @ast_to_expr.register
    def _(root: ast.Name):
        return expression.free_var(root.id)

    try:
        # this is really fragile, only works for functions that really stand alone
        source_code = inspect.getsource(func)
        return ast_to_expr(ast.parse(source_code).body[0])
    except Exception:
        print(f"Error transpiling ast: {traceback.format_exc()}")
        return None
