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
import polars as pl
import pytest

import bigframes as bf
import bigframes.core.compile.polars.compiler as polars_compiler
import bigframes.core.nodes as nodes
import bigframes.operations.json_ops as json_ops


def test_polars_to_json_string():
    """Test ToJSONString operation in Polars compiler."""
    compiler = polars_compiler.PolarsExpressionCompiler()
    op = json_ops.ToJSONString()
    # Polars doesn't have a native JSON type, it uses strings.
    # The operation is a cast to string.
    input_expr = pl.lit('{"b": 2}', dtype=pl.String)
    result = compiler.compile_op(op, input_expr)

    df = pl.DataFrame({"a": ['{"b": 2}']}).lazy()
    result_df = df.with_columns(result.alias("b")).collect()
    assert result_df["b"][0] == '{"b": 2}'
    assert result_df["b"].dtype == pl.String


def test_polars_parse_json():
    """Test ParseJSON operation in Polars compiler."""
    compiler = polars_compiler.PolarsExpressionCompiler()
    op = json_ops.ParseJSON()
    input_expr = pl.lit('{"b": 2}', dtype=pl.String)
    result = compiler.compile_op(op, input_expr)

    df = pl.DataFrame({"a": ['{"b": 2}']}).lazy()
    result_df = df.with_columns(result.alias("b")).collect()
    # The result of json_decode is a struct
    assert isinstance(result_df["b"][0], dict)
    assert result_df["b"][0] == {"b": 2}


@pytest.mark.skip(reason="Polars does not have json_extract on string expressions")
def test_polars_json_extract():
    """Test JSONExtract operation in Polars compiler."""
    compiler = polars_compiler.PolarsExpressionCompiler()
    op = json_ops.JSONExtract(json_path="$.b")
    input_expr = pl.lit('{"a": 1, "b": "hello"}', dtype=pl.String)
    result = compiler.compile_op(op, input_expr)

    df = pl.DataFrame({"a": ['{"b": "world"}']}).lazy()
    result_df = df.with_columns(result.alias("b")).collect()
    # json_extract returns a JSON encoded string
    assert result_df["b"][0] == '"world"'


def test_readlocal_with_json_column(polars_session):
    """Test ReadLocalNode compilation with JSON columns."""
    pandas_df = pd.DataFrame({"data": ['{"key": "value"}']})
    pandas_df["data"] = pandas_df["data"].astype(bf.dtypes.JSON_DTYPE)
    bf_df = polars_session.read_pandas(pandas_df)

    node = bf_df._block.expr.node
    # Traverse the node tree to find the ReadLocalNode
    while not isinstance(node, nodes.ReadLocalNode):
        node = node.child
    assert isinstance(node, nodes.ReadLocalNode)

    compiler = polars_compiler.PolarsCompiler()
    lazy_frame = compiler.compile_node(node)
    result_df = lazy_frame.collect()

    # The compiler should have converted the JSON column to string.
    assert result_df.schema["column_0"] == pl.String
    assert result_df["column_0"][0] == '{"key":"value"}'
