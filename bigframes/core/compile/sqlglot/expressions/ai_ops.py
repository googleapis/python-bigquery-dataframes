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

from dataclasses import asdict
import typing

import sqlglot.expressions as sge

from bigframes import operations as ops
from bigframes.core.compile.sqlglot import scalar_compiler
from bigframes.core.compile.sqlglot.expressions.typed_expr import TypedExpr

register_nary_op = scalar_compiler.scalar_op_compiler.register_nary_op


@register_nary_op(ops.AIGenerateBool, pass_op=True)
def _(*exprs: TypedExpr, op: ops.AIGenerateBool) -> sge.Expression:
    args = [_construct_prompt(exprs, op.prompt_context)] + _construct_named_args(op)

    return sge.func("AI.GENERATE_BOOL", *args)


@register_nary_op(ops.AIGenerateInt, pass_op=True)
def _(*exprs: TypedExpr, op: ops.AIGenerateInt) -> sge.Expression:
    args = [_construct_prompt(exprs, op.prompt_context)] + _construct_named_args(op)

    return sge.func("AI.GENERATE_INT", *args)


def _construct_prompt(
    exprs: tuple[TypedExpr, ...], prompt_context: tuple[str | None, ...]
) -> sge.Kwarg:
    prompt: list[str | sge.Expression] = []
    column_ref_idx = 0

    for elem in prompt_context:
        if elem is None:
            prompt.append(exprs[column_ref_idx].expr)
        else:
            prompt.append(sge.Literal.string(elem))

    return sge.Kwarg(this="prompt", expression=sge.Tuple(expressions=prompt))


def _construct_named_args(op: ops.NaryOp) -> list[sge.Kwarg]:
    args = []

    op_args = asdict(op)

    connection_id = typing.cast(str, op_args["connection_id"])
    args.append(
        sge.Kwarg(this="connection_id", expression=sge.Literal.string(connection_id))
    )

    endpoit = typing.cast(str, op_args.get("endpoint", None))
    if endpoit is not None:
        args.append(sge.Kwarg(this="endpoint", expression=sge.Literal.string(endpoit)))

    request_type = typing.cast(str, op_args["request_type"]).upper()
    args.append(
        sge.Kwarg(this="request_type", expression=sge.Literal.string(request_type))
    )

    model_params = typing.cast(str, op_args.get("model_params", None))
    if model_params is not None:
        args.append(
            sge.Kwarg(
                this="model_params",
                # sge.JSON requires the SQLGlot version to be at least 25.18.0
                # PARSE_JSON won't work as the function requires a JSON literal.
                expression=sge.JSON(this=sge.Literal.string(model_params)),
            )
        )

    return args
