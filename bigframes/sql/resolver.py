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

import functools
from typing import TYPE_CHECKING

import google.cloud.bigquery

from bigframes.core import array_value, bigframe_node, expression, nodes
import bigframes.dtypes

if TYPE_CHECKING:
    import bigframes.session


# TODO: Handle aggregate, windows sql expressions
def resolve_sql_exprs(
    root: bigframe_node.BigFrameNode, session: bigframes.session.Session
) -> bigframe_node.BigFrameNode:
    def resolve_node(node: bigframe_node.BigFrameNode) -> bigframe_node.BigFrameNode:
        def resolve_expr(expr: expression.Expression, input_node: nodes.UnaryNode):
            def resolve_expr_step(expr: expression.Expression) -> expression.Expression:
                if isinstance(expr, expression.RawSqlExpression):
                    return resolve_raw_sql(expr, input_node.child, session)
                return expr

            return resolve_expr_step(expr.transform_children(resolve_expr_step))

        if isinstance(
            node, (nodes.ProjectionNode, nodes.FilterNode, nodes.OrderByNode)
        ):
            return node.transform_exprs(
                functools.partial(resolve_expr, input_node=node)
            )
        else:
            return node

    return root.bottom_up(resolve_node)


@functools.cache
def resolve_raw_sql(
    expr: expression.RawSqlExpression,
    input_node: nodes.BigFrameNode,
    session: bigframes.session.Session,
):
    inputs_sql = session._executor.to_sql(array_value.ArrayValue(input_node))
    as_sql = f"SELECT {expr.sql} from ({inputs_sql})"
    # print(as_sql)
    bqclient = session.bqclient
    job = bqclient.query(
        as_sql, job_config=google.cloud.bigquery.QueryJobConfig(dry_run=True)
    )
    _, output_type = bigframes.dtypes.convert_schema_field(job.schema[0])
    context = expression.SqlFragmentContext(
        dependencies=tuple(id.sql for id in input_node.ids), output_type=output_type
    )
    return expression.RawSqlExpression(sql=expr.sql, context=context)
