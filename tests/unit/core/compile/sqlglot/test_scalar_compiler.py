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
from sqlglot import expressions as sge

from bigframes import dtypes
from bigframes.core import expression, field, identifiers
from bigframes.core.compile.sqlglot import scalar_compiler


def test_compile_deref():
    col_id = identifiers.ColumnId("my_col")
    field_by_id = {col_id: field.Field(col_id, dtypes.INT_DTYPE)}
    expr = expression.DerefOp(col_id)

    result = scalar_compiler.compile_deref_op(expr, field_by_id)

    assert result == sge.ColumnDef(this=sge.to_identifier(col_id.sql, quoted=True))


def test_compile_deref_with_dispatched_function_raise_error():
    expr = expression.deref("my_col")

    with pytest.raises(ValueError):
        scalar_compiler.compile_scalar_expression(expr)
