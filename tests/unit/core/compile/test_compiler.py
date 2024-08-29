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

import google.cloud.bigquery as bq
import pytest
import re

from bigframes.core.compile.compiler import Compiler
from bigframes.core.nodes import DqlStatementNode



@pytest.mark.parametrize(
    "strict",
    [pytest.param(True), pytest.param(False)]

)
def test_compile_dql_statement_node(strict: bool):
    compiler = Compiler(strict)
    node = DqlStatementNode("SELECT a,b FROM MyTable", (bq.SchemaField("a", "INTEGER"), bq.SchemaField("b", "FLOAT")))

    result = compiler.compile_node(node)

    assert re.sub('\s+', ' ', result.to_sql()) == "SELECT t0.`a`, t0.`b` FROM ( SELECT a, b FROM MyTable ) AS t0"

