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

import pandas as pd
import pytest

import bigframes.bigquery as bbq
import bigframes.operations.search_ops as search_ops
import bigframes.series
import bigframes.session
import bigframes.testing.mocks


@pytest.fixture
def mock_session():
    return bigframes.testing.mocks.create_bigquery_session()


def test_search_series(mock_session):
    # Use real Series backed by mock session (via read_pandas/ReadLocalNode)
    s = bigframes.series.Series(["foo bar", "baz"], session=mock_session)
    search_query = "foo"
    result = bbq.search(s, search_query)

    # Verify the operation in the expression tree
    import bigframes.core.nodes as nodes
    import bigframes.core.expression as ex

    # Get the underlying node
    node = result._block.expr.node

    # Traverse down to find the ProjectionNode
    while isinstance(node, nodes.SelectionNode):
        node = node.child

    # It should be a ProjectionNode (since search is a unary op applied to existing data)
    assert isinstance(node, nodes.ProjectionNode)

    # Find the assignment corresponding to the result column
    # result._value_column corresponds to one of the output columns of the SelectionNode chain
    # But checking the ProjectionNode assignments directly is easier if we iterate through them.
    # The SearchOp should be one of the assignments.

    # Locate the assignment with SearchOp
    assignments = [expr for expr, id in node.assignments if isinstance(expr, ex.OpExpression) and isinstance(expr.op, search_ops.SearchOp)]
    assert len(assignments) == 1
    assignment = assignments[0]

    # The expression should be an OpExpression with SearchOp
    assert isinstance(assignment, ex.OpExpression)
    assert isinstance(assignment.op, search_ops.SearchOp)

    assert assignment.op.search_query == search_query
    assert assignment.op.json_scope is None
    assert assignment.op.analyzer is None
    assert assignment.op.analyzer_options is None


def test_search_series_with_options(mock_session):
    s = bigframes.series.Series(["foo bar", "baz"], session=mock_session)
    search_query = "foo"
    result = bbq.search(
        s,
        search_query,
        json_scope="JSON_VALUES",
        analyzer="LOG_ANALYZER",
        analyzer_options='{"delimiters": [" "]}',
    )

    # Verify the operation in the expression tree
    import bigframes.core.nodes as nodes
    import bigframes.core.expression as ex

    # Get the underlying node
    node = result._block.expr.node

    # Traverse down to find the ProjectionNode
    while isinstance(node, nodes.SelectionNode):
        node = node.child

    # It should be a ProjectionNode
    assert isinstance(node, nodes.ProjectionNode)

    # Locate the assignment with SearchOp
    assignments = [expr for expr, id in node.assignments if isinstance(expr, ex.OpExpression) and isinstance(expr.op, search_ops.SearchOp)]
    assert len(assignments) == 1
    assignment = assignments[0]

    assert isinstance(assignment, ex.OpExpression)
    assert isinstance(assignment.op, search_ops.SearchOp)

    assert assignment.op.search_query == search_query
    assert assignment.op.json_scope == "JSON_VALUES"
    assert assignment.op.analyzer == "LOG_ANALYZER"
    assert assignment.op.analyzer_options == '{"delimiters": [" "]}'


def test_search_dataframe(mock_session):
    # Mock dataframe with 2 columns
    df = pd.DataFrame({"col1": ["foo", "bar"], "col2": ["baz", "qux"]})
    bf = bigframes.dataframe.DataFrame(df, session=mock_session)

    search_query = "foo"
    result = bbq.search(bf, search_query)

    import bigframes.core.nodes as nodes
    import bigframes.core.expression as ex
    from bigframes.operations import struct_ops

    # Get the underlying node
    node = result._block.expr.node

    # Traverse down to find the ProjectionNode
    while isinstance(node, nodes.SelectionNode):
        node = node.child

    # Should be a ProjectionNode
    assert isinstance(node, nodes.ProjectionNode)

    assignments = [expr for expr, id in node.assignments if isinstance(expr, ex.OpExpression) and isinstance(expr.op, search_ops.SearchOp)]
    assert len(assignments) == 1
    assignment = assignments[0]

    assert isinstance(assignment, ex.OpExpression)
    assert isinstance(assignment.op, search_ops.SearchOp)
    assert assignment.op.search_query == search_query

    # Verify that the input to SearchOp is a StructOp
    # The input expression to SearchOp
    search_input = assignment.inputs[0]

    # Since struct() op and search op might be in the same ProjectionNode or different ones.
    # If they are in the same ProjectionNode, `search_input` would be a DerefOp to a column not in assignments?
    # No, ProjectionNode assignments are parallel. So struct op must be in a child node.

    # Check if struct op is in the same node (unlikely for parallel projection unless merged somehow, but typical flow puts them sequential)

    # If search_input is DerefOp, we look in the child node.
    assert isinstance(search_input, ex.DerefOp)

    child_node = node.child
    # Traverse SelectionNodes if any
    while isinstance(child_node, nodes.SelectionNode):
        child_node = child_node.child

    # It should be a ProjectionNode (from struct())
    assert isinstance(child_node, nodes.ProjectionNode)

    # Find the struct assignment
    struct_col_id = search_input.id
    struct_assignment = next(expr for expr, id in child_node.assignments if id == struct_col_id)

    assert isinstance(struct_assignment, ex.OpExpression)
    assert isinstance(struct_assignment.op, struct_ops.StructOp)
    assert struct_assignment.op.column_names == ("col1", "col2")


def test_search_invalid_input(mock_session):
    with pytest.raises(ValueError, match="data_to_search must be a Series or DataFrame"):
        bbq.search("invalid", "foo")
