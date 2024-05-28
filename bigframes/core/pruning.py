# Copyright 2024 Google LLC
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

from typing import Sequence

import bigframes.core.expression as ex
import bigframes.operations as ops

COMPARISON_OP_TYPES = tuple(
    type(i)
    for i in (
        ops.eq_op,
        ops.eq_null_match_op,
        ops.ne_op,
        ops.gt_op,
        ops.ge_op,
        ops.lt_op,
        ops.le_op,
    )
)


def cluster_cols_for_predicate(predicate: ex.Expression) -> Sequence[str]:
    """Try to determine cluster col candidates that work with given predicates."""
    if isinstance(predicate, ex.UnboundVariableExpression):
        return [predicate.id]
    if isinstance(predicate, ex.OpExpression):
        op = predicate.op
        if isinstance(op, COMPARISON_OP_TYPES):
            return cluster_cols_for_comparison(predicate.inputs[0], predicate.inputs[1])
        if isinstance(op, (type(ops.invert_op))):
            return cluster_cols_for_predicate(predicate.inputs[0])
        if isinstance(op, (type(ops.and_op), type(ops.or_op))):
            left_cols = cluster_cols_for_predicate(predicate.inputs[0])
            right_cols = cluster_cols_for_predicate(predicate.inputs[1])
            return [*left_cols, *[col for col in right_cols if col not in left_cols]]
        else:
            return []
    else:
        # Constant
        return []


def cluster_cols_for_comparison(
    left_ex: ex.Expression, right_ex: ex.Expression
) -> Sequence[str]:
    if left_ex.is_const:
        if isinstance(right_ex, ex.UnboundVariableExpression):
            return [right_ex.id]
    elif right_ex.is_const:
        if isinstance(left_ex, ex.UnboundVariableExpression):
            return [left_ex.id]
    return []
