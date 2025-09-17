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

from bigframes.core import expression as expr
import bigframes.pandas as bpd


def _apply_unary_ops(
    obj: bpd.DataFrame,
    ops_list: typing.Sequence[expr.Expression],
    new_names: typing.Sequence[str],
) -> str:
    array_value = obj._block.expr
    result, old_names = array_value.compute_values(ops_list)

    # Rename columns for deterministic golden SQL results.
    assert len(old_names) == len(new_names)
    col_ids = {old_name: new_name for old_name, new_name in zip(old_names, new_names)}
    result = result.rename_columns(col_ids).select_columns(new_names)

    sql = result.session._executor.to_sql(result, enable_cache=False)
    return sql
