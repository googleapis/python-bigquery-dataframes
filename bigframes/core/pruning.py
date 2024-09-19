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

import bigframes.core.expression as ex
import bigframes.core.identifiers as ids
import bigframes.core.schema as schemata
import bigframes.dtypes
import bigframes.operations as ops

LOW_CARDINALITY_TYPES = [bigframes.dtypes.BOOL_DTYPE]

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


def cluster_cols_for_predicate(
    predicate: ex.Expression, schema: schemata.ArraySchema
) -> list[ids.Identifier]:
    """Try to determine cluster col candidates that work with given predicates."""
    # TODO: Prioritize based on predicted selectivity (eg. equality conditions are probably very selective)
    if isinstance(predicate, ex.DerefExpression):
        val = schema.resolve_ref(predicate.ref)
        return [val.column] if bigframes.dtypes.is_clusterable(val.dtype) else []
    elif isinstance(predicate, ex.OpExpression):
        op = predicate.op
        # TODO: Support geo predicates, which support pruning if clustered (other than st_disjoint)
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/geography_functions
        if isinstance(op, COMPARISON_OP_TYPES):
            return cluster_cols_for_comparison(
                predicate.inputs[0], predicate.inputs[1], schema
            )
        elif isinstance(op, (type(ops.invert_op))):
            return cluster_cols_for_predicate(predicate.inputs[0], schema)
        elif isinstance(op, (type(ops.and_op), type(ops.or_op))):
            left_cols = cluster_cols_for_predicate(predicate.inputs[0], schema)
            right_cols = cluster_cols_for_predicate(predicate.inputs[1], schema)
            return [*left_cols, *[col for col in right_cols if col not in left_cols]]
    return []


def cluster_cols_for_comparison(
    left_ex: ex.Expression, right_ex: ex.Expression, schema: schemata.ArraySchema
) -> list[ids.Identifier]:
    # TODO: Try to normalize expressions such that one side is a single variable.
    # eg. Convert -cola>=3 to cola<-3 and colb+3 < 4 to colb < 1
    if left_ex.is_const:
        # There are some invertible ops that would also be ok
        if isinstance(right_ex, ex.DerefExpression):
            val = schema.resolve_ref(right_ex.ref)
            return [val.column] if bigframes.dtypes.is_clusterable(val.dtype) else []
    elif right_ex.is_const:
        if isinstance(left_ex, ex.DerefExpression):
            val = schema.resolve_ref(left_ex.ref)
            return [val.column] if bigframes.dtypes.is_clusterable(val.dtype) else []
    return []
