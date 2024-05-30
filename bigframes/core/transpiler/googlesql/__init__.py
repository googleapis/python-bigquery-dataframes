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

"""Python classes for GoogleSQL queries, adhering to the official syntax rules: 
https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#sql_syntax"""


from bigframes.core.transpiler.googlesql.cte import NonRecursiveCTE
from bigframes.core.transpiler.googlesql.expression import (
    AsAlias,
    Expression,
    StarExpression,
    TableRef,
    ColumnExpression,
)
from bigframes.core.transpiler.googlesql.from_ import FromClause, FromItem
from bigframes.core.transpiler.googlesql.functions import Count
from bigframes.core.transpiler.googlesql.join import (
    JoinCondition,
    JoinOperation,
    JoinOperator,
)
from bigframes.core.transpiler.googlesql.query import OrderByExpr, QueryExpr
from bigframes.core.transpiler.googlesql.select import (
    Select,
    SelectExpression,
)

__all__ = [
    "AsAlias",
    "Expression",
    "FromClause",
    "FromItem",
    "JoinCondition",
    "JoinOperation",
    "JoinOperator",
    "NonRecursiveCTE",
    "OrderByExpr",
    "QueryExpr",
    "Select",
    "SelectExpression",
    "TableRef",
    "Count",
    "StarExpression",
    "ColumnExpression",
]
