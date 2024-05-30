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

import dataclasses

from bigframes.core.transpiler.googlesql.query import QueryExpr
from bigframes.core.transpiler.googlesql.sql import SQLSyntax


@dataclasses.dataclass
class NonRecursiveCTE(SQLSyntax):
    """GoogleSQL non_recursive_cte syntax."""

    cte_name: str
    query_expr: QueryExpr

    def sql(self) -> str:
        return f"{self.cte_name} AS (\n{self.query_expr.sql()}\n)"
