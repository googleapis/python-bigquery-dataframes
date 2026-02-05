# Copyright 2023 Google LLC
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

import dataclasses
import functools
from typing import Mapping, Optional, Sequence, Tuple

from bigframes.core import identifiers, nodes
import bigframes.core.expression as ex
from bigframes.core.ordering import OrderingExpression
import bigframes.dtypes

# A fixed number of variable to assume for overhead on some operations
OVERHEAD_VARIABLES = 5


@dataclasses.dataclass(frozen=True, eq=True)
class ColumnDef:
    expression: ex.Expression
    id: identifiers.ColumnId


# TODO: Raw data source node, join node, union node


@dataclasses.dataclass(frozen=True)
class SelectNode(nodes.UnaryNode):
    selections: tuple[nodes.ColumnDef, ...] = ()
    predicates: tuple[ex.Expression, ...] = ()
    sorting: tuple[OrderingExpression, ...] = ()
    limit: Optional[int] = None

    def __post_init__(self):
        try:
            self.fields
        except Exception:
            ...

    @functools.cached_property
    def fields(self) -> Sequence[nodes.Field]:
        fields = []
        for cdef in self.selections:
            bound_expr = ex.bind_schema_fields(cdef.expression, self.child.field_by_id)
            field = nodes.Field(
                cdef.id,
                bigframes.dtypes.dtype_for_etype(bound_expr.output_type),
                nullable=bound_expr.nullable,
            )

            # Special case until we get better nullability inference in expression objects themselves
            if bound_expr.is_identity and not any(
                self.child.field_by_id[id].nullable
                for id in cdef.expression.column_references
            ):
                field = field.with_nonnull()
            fields.append(field)

        return tuple(fields)

    @property
    def variables_introduced(self) -> int:
        # This operation only renames variables, doesn't actually create new ones
        return 0

    @property
    def defines_namespace(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        if self.child.row_count is not None:
            if self.limit is not None:
                return min([self.limit, self.child.row_count])
            return self.child.row_count

        return None

    @property
    def node_defined_ids(self) -> Tuple[identifiers.ColumnId, ...]:
        return tuple(cdef.id for cdef in self.selections)

    @property
    def consumed_ids(self):
        raise NotImplementedError()

    @property
    def _node_expressions(self):
        raise NotImplementedError()

    @functools.cache
    def get_id_mapping(self) -> dict[identifiers.ColumnId, ex.Expression]:
        return {cdef.id: cdef.expression for cdef in self.selections}

    def remap_vars(
        self, mappings: Mapping[identifiers.ColumnId, identifiers.ColumnId]
    ) -> SelectNode:
        raise NotImplementedError()

    def remap_refs(
        self, mappings: Mapping[identifiers.ColumnId, identifiers.ColumnId]
    ) -> SelectNode:
        raise NotImplementedError()  # type: ignore
