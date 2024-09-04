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
import pandas as pd

from bigframes.core.nodes import DqlStatementNode
from bigframes.core.schema import ArraySchema, SchemaItem


def test_dql_statement_node():
    node = DqlStatementNode(
        sql="SELECT a,b FROM Table1 INNER JOIN Table2 USING c",
        physical_schema=(bq.SchemaField("a", "INTEGER"), bq.SchemaField("b", "FLOAT")),
        referenced_table_count=2,
    )

    assert node.schema == ArraySchema(
        (SchemaItem("a", pd.Int64Dtype()), SchemaItem("b", pd.Float64Dtype()))
    )
    assert node.variables_introduced == 12
    assert node.explicitly_ordered is False
    assert node.order_ambiguous is True
