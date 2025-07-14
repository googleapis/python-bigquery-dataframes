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

from typing import TYPE_CHECKING

import bigframes.core.compile.polars.compiler as polars_compiler
from bigframes.operations.generic_ops import notnull_op

if TYPE_CHECKING:
    import polars as pl


def _polars_notnull_op_impl(
    compiler: polars_compiler.PolarsExpressionCompiler,
    op: notnull_op.NotNullOp,
    input: pl.Expr,
) -> pl.Expr:
    return input.is_not_null()


if hasattr(polars_compiler, "PolarsExpressionCompiler"):
    # TODO(https://github.com/python/mypy/issues/13040): remove `type: ignore`
    # when mypy can better handle singledispatch.
    polars_compiler.PolarsExpressionCompiler.compile_op.register(  # type: ignore
        notnull_op.NotNullOp, _polars_notnull_op_impl
    )
