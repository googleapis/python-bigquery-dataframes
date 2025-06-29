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

import pytest
import sqlglot.expressions as sge

import bigframes.core.expression as expr
import bigframes.core.nodes as nodes
from bigframes.core.compile.sqlglot import compiler
import bigframes.core.compile.sqlglot.sqlglot_ir as ir
import bigframes.core.compile.sqlglot.scalar_compiler as scalar_compiler
from bigframes.core.compile.sqlglot.compiler import SQLGlotCompiler
import bigframes.dtypes as dtypes
from bigframes.core.identifiers import ColumnId, ExpressionId


@pytest.fixture
def sqlglot_compiler():
    return SQLGlotCompiler()

@pytest.fixture
def base_ir(sqlglot_compiler: SQLGlotCompiler):
    # Creates a simple base IR: SELECT col1, col2 FROM table_name
    return ir.SQLGlotIR.from_table(
        "project_id", "dataset_id", "table_name",
        col_names=["col1", "col2"],
        alias_names=["col1_alias", "col2_alias"],
        uid_gen=sqlglot_compiler.uid_gen
    )

def test_compile_isin_simple(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test basic IS IN functionality."""
    col_id = ColumnId("col1_alias")
    values = [
        expr.LiteralExpression(1, dtype=dtypes.INT_DTYPE),
        expr.LiteralExpression(2, dtype=dtypes.INT_DTYPE),
    ]
    isin_node = nodes.IsInNode(
        child=None,  # Child is not used by _compile_node directly
        column_id=col_id,
        values=tuple(values),
        negate=False
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name" WHERE "col1_alias" IN (1, 2)'
    assert compiled_ir.sql == expected_sql

def test_compile_isin_negate(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test IS NOT IN functionality."""
    col_id = ColumnId("col2_alias")
    values = [
        expr.LiteralExpression("a", dtype=dtypes.STRING_DTYPE),
        expr.LiteralExpression("b", dtype=dtypes.STRING_DTYPE),
    ]
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id,
        values=tuple(values),
        negate=True
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name" WHERE "col2_alias" NOT IN (\'a\', \'b\')'
    assert compiled_ir.sql == expected_sql

def test_compile_isin_empty_values_negate_false(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test IS IN with an empty list of values (should always be false)."""
    col_id = ColumnId("col1_alias")
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id,
        values=tuple(),
        negate=False
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    # SQL standard for 'X IN ()' is false.
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name" WHERE FALSE'
    assert compiled_ir.sql == expected_sql

def test_compile_isin_empty_values_negate_true(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test IS NOT IN with an empty list of values (should always be true)."""
    col_id = ColumnId("col1_alias")
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id,
        values=tuple(),
        negate=True
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    # SQL standard for 'X NOT IN ()' is true.
    # The implementation returns the original IR if negate is true and values are empty.
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name"'
    assert compiled_ir.sql == expected_sql

def test_compile_isin_different_types(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test IS IN with different data types."""
    col_id = ColumnId("col1_alias")
    values = [
        expr.LiteralExpression(1.0, dtype=dtypes.FLOAT_DTYPE),
        expr.LiteralExpression(True, dtype=dtypes.BOOL_DTYPE),
        # Note: Mixing types in an IN clause might have specific database behaviors,
        # but the compilation should still produce valid SQL.
    ]
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id,
        values=tuple(values),
        negate=False
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name" WHERE "col1_alias" IN (1.0, TRUE)'
    assert compiled_ir.sql == expected_sql

def test_compile_isin_with_non_literal_values(sqlglot_compiler: SQLGlotCompiler, base_ir: ir.SQLGlotIR):
    """Test IS IN with non-literal expressions (e.g., another column or function)."""
    # This test assumes scalar_compiler can handle compiling various expression types.
    col_id_check = ColumnId("col1_alias")
    col_id_val1 = ColumnId("col2_alias") # Value from another column

    # Example of a more complex expression, e.g. an operation
    # For simplicity, using another literal here, but could be a function call
    # This requires ExpressionId to be defined for such expressions if they are part of the IR.
    # For now, using literals to represent what these expressions would compile to.
    # This part of the test might need adjustment based on how Expression and IsInNode are typically constructed.

    # Let's assume values can be simple literals for this test structure,
    # as `IsInNode.values` expects `ScalarExpression`.
    # A more complex scenario would involve `IsInNode` being part of a larger expression tree.
    values = [
        expr.DerefOp(col_id_val1), # Check if col1_alias is in values from col2_alias
        expr.LiteralExpression(100, dtype=dtypes.INT_DTYPE)
    ]
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id_check,
        values=tuple(values),
        negate=False
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir)
    # The DerefOp for col_id_val1 should compile to its SQL name "col2_alias"
    expected_sql = 'SELECT "col1_alias", "col2_alias" FROM "project_id"."dataset_id"."table_name" WHERE "col1_alias" IN ("col2_alias", 100)'
    assert compiled_ir.sql == expected_sql

# Future test ideas:
# - Test with NULL values in the list (if supported/defined behavior).
# - Test with NULLs in the column being checked.
# - Test with a subquery in the IN clause (though IsInNode might not directly support this,
#   it's a common SQL pattern that might be relevant for future extensions).
# - Test type coercion scenarios if applicable.
# - Test with column names that need quoting (e.g., contain spaces or special chars).
#   The current base_ir and ColumnId setup might implicitly handle this via sqlglot's quoting.
#   `scalar_compiler.compile_scalar_expression(expression.DerefOp(node.column_id))`
#   should correctly quote if `node.column_id.sql` gives a name needing quotes.
#   And `ir.SQLGlotIR.from_table` also handles quoting for its alias_names.
#   So, this might be covered, but an explicit test could be valuable.

# Example of a column that might need quoting
@pytest.fixture
def base_ir_quoting(sqlglot_compiler: SQLGlotCompiler):
    return ir.SQLGlotIR.from_table(
        "project", "dataset", "table",
        col_names=["col with space", "another col"],
        alias_names=["col with space_alias", "another col_alias"], # sqlglot will quote these
        uid_gen=sqlglot_compiler.uid_gen
    )

def test_compile_isin_quoting(sqlglot_compiler: SQLGlotCompiler, base_ir_quoting: ir.SQLGlotIR):
    """Test IS IN with column names that require quoting."""
    col_id = ColumnId("col with space_alias")
    values = [
        expr.LiteralExpression("value1", dtype=dtypes.STRING_DTYPE),
    ]
    isin_node = nodes.IsInNode(
        child=None,
        column_id=col_id,
        values=tuple(values),
        negate=False
    )
    compiled_ir = sqlglot_compiler._compile_node(isin_node, base_ir_quoting)
    # Expecting "col with space_alias" to be quoted by sqlglot
    expected_sql = 'SELECT "col with space_alias", "another col_alias" FROM "project"."dataset"."table" WHERE "col with space_alias" IN (\'value1\')'
    assert compiled_ir.sql == expected_sql
