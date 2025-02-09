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

import abc
import dataclasses
import datetime
import functools
import itertools
import typing
from typing import Callable, cast, Iterable, Mapping, Optional, Sequence, Tuple, TypeVar

import google.cloud.bigquery as bq

import bigframes.core.expression as ex
import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.core.identifiers as bfet_ids
from bigframes.core.ordering import OrderingExpression
import bigframes.core.schema as schemata
import bigframes.core.slices as slices
import bigframes.core.window_spec as window
import bigframes.dtypes

if typing.TYPE_CHECKING:
    import bigframes.core.ordering as orderings
    import bigframes.session


# A fixed number of variable to assume for overhead on some operations
OVERHEAD_VARIABLES = 5

COLUMN_SET = frozenset[bfet_ids.ColumnId]

Self = TypeVar("Self")


@dataclasses.dataclass(frozen=True)
class Field:
    id: bfet_ids.ColumnId
    dtype: bigframes.dtypes.Dtype


@dataclasses.dataclass(eq=False, frozen=True)
class BigFrameNode(abc.ABC):
    """
    Immutable node for representing 2D typed array as a tree of operators.

    All subclasses must be hashable so as to be usable as caching key.
    """

    @property
    def deterministic(self) -> bool:
        """Whether this node will evaluates deterministically."""
        return True

    @property
    def row_preserving(self) -> bool:
        """Whether this node preserves input rows."""
        return True

    @property
    def non_local(self) -> bool:
        """
        Whether this node combines information across multiple rows instead of processing rows independently.
        Used as an approximation for whether the expression may require shuffling to execute (and therefore be expensive).
        """
        return False

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        """Direct children of this node"""
        return tuple([])

    @property
    @abc.abstractmethod
    def row_count(self) -> typing.Optional[int]:
        return None

    @abc.abstractmethod
    def remap_vars(
        self: Self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> Self:
        """Remap defined (in this node only) variables."""
        ...

    @abc.abstractmethod
    def remap_refs(
        self: Self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> Self:
        """Remap variable references"""
        ...

    @property
    @abc.abstractmethod
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        """The variables defined in this node (as opposed to by child nodes)."""
        ...

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset()

    @functools.cached_property
    def session(self):
        sessions = []
        for child in self.child_nodes:
            if child.session is not None:
                sessions.append(child.session)
        unique_sessions = len(set(sessions))
        if unique_sessions > 1:
            raise ValueError("Cannot use combine sources from multiple sessions.")
        elif unique_sessions == 1:
            return sessions[0]
        return None

    def _validate(self):
        """Validate the local data in the node."""
        return

    @functools.cache
    def validate_tree(self) -> bool:
        for child in self.child_nodes:
            child.validate_tree()
        self._validate()
        field_list = list(self.fields)
        if len(set(field_list)) != len(field_list):
            raise ValueError(f"Non unique field ids {list(self.fields)}")
        return True

    def _as_tuple(self) -> Tuple:
        """Get all fields as tuple."""
        return tuple(getattr(self, field.name) for field in dataclasses.fields(self))

    def __hash__(self) -> int:
        # Custom hash that uses cache to avoid costly recomputation
        return self._cached_hash

    def __eq__(self, other) -> bool:
        # Custom eq that tries to short-circuit full structural comparison
        if not isinstance(other, self.__class__):
            return False
        if self is other:
            return True
        if hash(self) != hash(other):
            return False
        return self._as_tuple() == other._as_tuple()

    # BigFrameNode trees can be very deep so its important avoid recalculating the hash from scratch
    # Each subclass of BigFrameNode should use this property to implement __hash__
    # The default dataclass-generated __hash__ method is not cached
    @functools.cached_property
    def _cached_hash(self):
        return hash(self._as_tuple())

    @property
    def roots(self) -> typing.Set[BigFrameNode]:
        roots = itertools.chain.from_iterable(
            map(lambda child: child.roots, self.child_nodes)
        )
        return set(roots)

    # TODO: Store some local data lazily for select, aggregate nodes.
    @property
    @abc.abstractmethod
    def fields(self) -> Iterable[Field]:
        ...

    @property
    def ids(self) -> Iterable[bfet_ids.ColumnId]:
        """All output ids from the node."""
        return (field.id for field in self.fields)

    @property
    @abc.abstractmethod
    def variables_introduced(self) -> int:
        """
        Defines number of values created by the current node. Helps represent the "width" of a query
        """
        ...

    @property
    def relation_ops_created(self) -> int:
        """
        Defines the number of relational ops generated by the current node. Used to estimate query planning complexity.
        """
        return 1

    @property
    def joins(self) -> bool:
        """
        Defines whether the node joins data.
        """
        return False

    @property
    @abc.abstractmethod
    def order_ambiguous(self) -> bool:
        """
        Whether row ordering is potentially ambiguous. For example, ReadTable (without a primary key) could be ordered in different ways.
        """
        ...

    @property
    @abc.abstractmethod
    def explicitly_ordered(self) -> bool:
        """
        Whether row ordering is potentially ambiguous. For example, ReadTable (without a primary key) could be ordered in different ways.
        """
        ...

    @functools.cached_property
    def height(self) -> int:
        if len(self.child_nodes) == 0:
            return 0
        return max(child.height for child in self.child_nodes) + 1

    @functools.cached_property
    def total_variables(self) -> int:
        return self.variables_introduced + sum(
            map(lambda x: x.total_variables, self.child_nodes)
        )

    @functools.cached_property
    def total_relational_ops(self) -> int:
        return self.relation_ops_created + sum(
            map(lambda x: x.total_relational_ops, self.child_nodes)
        )

    @functools.cached_property
    def total_joins(self) -> int:
        return int(self.joins) + sum(map(lambda x: x.total_joins, self.child_nodes))

    @functools.cached_property
    def schema(self) -> schemata.ArraySchema:
        # TODO: Make schema just a view on fields
        return schemata.ArraySchema(
            tuple(schemata.SchemaItem(i.id.name, i.dtype) for i in self.fields)
        )

    @property
    def planning_complexity(self) -> int:
        """
        Empirical heuristic measure of planning complexity.

        Used to determine when to decompose overly complex computations. May require tuning.
        """
        return self.total_variables * self.total_relational_ops * (1 + self.total_joins)

    @abc.abstractmethod
    def transform_children(
        self: Self, t: Callable[[BigFrameNode], BigFrameNode]
    ) -> Self:
        """Apply a function to each child node."""
        ...

    @property
    def defines_namespace(self) -> bool:
        """
        If true, this node establishes a new column id namespace.

        If false, this node consumes and produces ids in the namespace
        """
        return False

    def get_type(self, id: bfet_ids.ColumnId) -> bigframes.dtypes.Dtype:
        return self._dtype_lookup[id]

    @functools.cached_property
    def _dtype_lookup(self):
        return {field.id: field.dtype for field in self.fields}


class AdditiveNode:
    """Definition of additive - if you drop added_fields, you end up with the descendent.

    .. code-block:: text

        AdditiveNode (fields: a, b, c; added_fields: c)
            |
            |  additive_base
            V
        BigFrameNode (fields: a, b)

    """

    @property
    @abc.abstractmethod
    def added_fields(self) -> Tuple[Field, ...]:
        ...

    @property
    @abc.abstractmethod
    def additive_base(self) -> BigFrameNode:
        ...

    @abc.abstractmethod
    def replace_additive_base(self, BigFrameNode):
        ...


@dataclasses.dataclass(frozen=True, eq=False)
class UnaryNode(BigFrameNode):
    child: BigFrameNode

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.child,)

    @property
    def fields(self) -> Iterable[Field]:
        return self.child.fields

    @property
    def explicitly_ordered(self) -> bool:
        return self.child.explicitly_ordered

    def transform_children(
        self, t: Callable[[BigFrameNode], BigFrameNode]
    ) -> UnaryNode:
        transformed = dataclasses.replace(self, child=t(self.child))
        if self == transformed:
            # reusing existing object speeds up eq, and saves a small amount of memory
            return self
        return transformed

    def replace_child(self, new_child: BigFrameNode) -> UnaryNode:
        new_self = dataclasses.replace(self, child=new_child)  # type: ignore
        return new_self

    @property
    def order_ambiguous(self) -> bool:
        return self.child.order_ambiguous


@dataclasses.dataclass(frozen=True, eq=False)
class SliceNode(UnaryNode):
    """Logical slice node conditionally becomes limit or filter over row numbers."""

    start: Optional[int]
    stop: Optional[int]
    step: int = 1

    @property
    def row_preserving(self) -> bool:
        """Whether this node preserves input rows."""
        return False

    @property
    def non_local(self) -> bool:
        """
        Whether this node combines information across multiple rows instead of processing rows independently.
        Used as an approximation for whether the expression may require shuffling to execute (and therefore be expensive).
        """
        return True

    # these are overestimates, more accurate numbers available by converting to concrete limit or analytic+filter ops
    @property
    def variables_introduced(self) -> int:
        return 2

    @property
    def relation_ops_created(self) -> int:
        return 2

    @property
    def is_limit(self) -> bool:
        """Returns whether this is equivalent to a ORDER BY ... LIMIT N."""
        # TODO: Handle tail case.
        return (
            (not self.start)
            and (self.step == 1)
            and (self.stop is not None)
            and (self.stop > 0)
        )

    @property
    def row_count(self) -> typing.Optional[int]:
        child_length = self.child.row_count
        if child_length is None:
            return None
        return slices.slice_output_rows(
            (self.start, self.stop, self.step), child_length
        )

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset()

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> SliceNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> SliceNode:
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class InNode(BigFrameNode, AdditiveNode):
    """
    Special Join Type that only returns rows from the left side, as well as adding a bool column indicating whether a match exists on the right side.

    Modelled separately from join node, as this operation preserves row identity.
    """

    left_child: BigFrameNode
    right_child: BigFrameNode
    left_col: ex.DerefOp
    right_col: ex.DerefOp
    indicator_col: bfet_ids.ColumnId

    def _validate(self):
        assert not (
            set(self.left_child.ids) & set(self.right_child.ids)
        ), "Join ids collide"

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def non_local(self) -> bool:
        return True

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.left_child, self.right_child)

    @property
    def order_ambiguous(self) -> bool:
        return False

    @property
    def explicitly_ordered(self) -> bool:
        # Preserves left ordering always
        return True

    @property
    def added_fields(self) -> Tuple[Field, ...]:
        return (Field(self.indicator_col, bigframes.dtypes.BOOL_DTYPE),)

    @property
    def fields(self) -> Iterable[Field]:
        return itertools.chain(
            self.left_child.fields,
            self.added_fields,
        )

    @functools.cached_property
    def variables_introduced(self) -> int:
        """Defines the number of variables generated by the current node. Used to estimate query planning complexity."""
        return 1

    @property
    def joins(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        return self.left_child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.indicator_col,)

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset({self.left_col.id, self.right_col.id})

    @property
    def additive_base(self) -> BigFrameNode:
        return self.left_child

    def replace_additive_base(self, node: BigFrameNode):
        return dataclasses.replace(self, left_child=node)

    def transform_children(self, t: Callable[[BigFrameNode], BigFrameNode]) -> InNode:
        transformed = dataclasses.replace(
            self, left_child=t(self.left_child), right_child=t(self.right_child)
        )
        if self == transformed:
            # reusing existing object speeds up eq, and saves a small amount of memory
            return self
        return transformed

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> InNode:
        return dataclasses.replace(
            self, indicator_col=mappings.get(self.indicator_col, self.indicator_col)
        )

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> InNode:
        return dataclasses.replace(self, left_col=self.left_col.remap_column_refs(mappings, allow_partial_bindings=True), right_col=self.right_col.remap_column_refs(mappings, allow_partial_bindings=True))  # type: ignore


@dataclasses.dataclass(frozen=True, eq=False)
class JoinNode(BigFrameNode):
    left_child: BigFrameNode
    right_child: BigFrameNode
    conditions: typing.Tuple[typing.Tuple[ex.DerefOp, ex.DerefOp], ...]
    type: typing.Literal["inner", "outer", "left", "right", "cross"]

    def _validate(self):
        assert not (
            set(self.left_child.ids) & set(self.right_child.ids)
        ), "Join ids collide"

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def non_local(self) -> bool:
        return True

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.left_child, self.right_child)

    @property
    def order_ambiguous(self) -> bool:
        return True

    @property
    def explicitly_ordered(self) -> bool:
        # Do not consider user pre-join ordering intent - they need to re-order post-join in unordered mode.
        return False

    @property
    def fields(self) -> Iterable[Field]:
        return itertools.chain(self.left_child.fields, self.right_child.fields)

    @functools.cached_property
    def variables_introduced(self) -> int:
        """Defines the number of variables generated by the current node. Used to estimate query planning complexity."""
        return OVERHEAD_VARIABLES

    @property
    def joins(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        if self.type == "cross":
            if self.left_child.row_count is None or self.right_child.row_count is None:
                return None
            return self.left_child.row_count * self.right_child.row_count

        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset(
            itertools.chain.from_iterable(
                (*l_cond.column_references, *r_cond.column_references)
                for l_cond, r_cond in self.conditions
            )
        )

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(*self.ids, *self.referenced_ids)

    def transform_children(self, t: Callable[[BigFrameNode], BigFrameNode]) -> JoinNode:
        transformed = dataclasses.replace(
            self, left_child=t(self.left_child), right_child=t(self.right_child)
        )
        if self == transformed:
            # reusing existing object speeds up eq, and saves a small amount of memory
            return self
        return transformed

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> JoinNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> JoinNode:
        new_conds = tuple(
            (
                l_cond.remap_column_refs(mappings, allow_partial_bindings=True),
                r_cond.remap_column_refs(mappings, allow_partial_bindings=True),
            )
            for l_cond, r_cond in self.conditions
        )
        return dataclasses.replace(self, conditions=new_conds)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=False)
class ConcatNode(BigFrameNode):
    # TODO: Explcitly map column ids from each child
    children: Tuple[BigFrameNode, ...]
    output_ids: Tuple[bfet_ids.ColumnId, ...]

    def _validate(self):
        if len(self.children) == 0:
            raise ValueError("Concat requires at least one input table. Zero provided.")
        child_schemas = [child.schema.dtypes for child in self.children]
        if not len(set(child_schemas)) == 1:
            raise ValueError("All inputs must have identical dtypes. {child_schemas}")

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return self.children

    @property
    def order_ambiguous(self) -> bool:
        return any(child.order_ambiguous for child in self.children)

    @property
    def explicitly_ordered(self) -> bool:
        # Consider concat as an ordered operations (even though input frames may not be ordered)
        return True

    @property
    def fields(self) -> Iterable[Field]:
        # TODO: Output names should probably be aligned beforehand or be part of concat definition
        return (
            Field(id, field.dtype)
            for id, field in zip(self.output_ids, self.children[0].fields)
        )

    @functools.cached_property
    def variables_introduced(self) -> int:
        """Defines the number of variables generated by the current node. Used to estimate query planning complexity."""
        return len(self.schema.items) + OVERHEAD_VARIABLES

    @property
    def row_count(self) -> Optional[int]:
        sub_counts = [node.row_count for node in self.child_nodes]
        total = 0
        for count in sub_counts:
            if count is None:
                return None
            total += count
        return total

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return self.output_ids

    def transform_children(
        self, t: Callable[[BigFrameNode], BigFrameNode]
    ) -> ConcatNode:
        transformed = dataclasses.replace(
            self, children=tuple(t(child) for child in self.children)
        )
        if self == transformed:
            # reusing existing object speeds up eq, and saves a small amount of memory
            return self
        return transformed

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ConcatNode:
        new_ids = tuple(mappings.get(id, id) for id in self.output_ids)
        return dataclasses.replace(self, output_ids=new_ids)

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ConcatNode:
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class FromRangeNode(BigFrameNode):
    # TODO: Enforce single-row, single column constraint
    start: BigFrameNode
    end: BigFrameNode
    step: int
    output_id: bfet_ids.ColumnId = bfet_ids.ColumnId("labels")

    @property
    def roots(self) -> typing.Set[BigFrameNode]:
        return {self}

    @property
    def child_nodes(self) -> typing.Sequence[BigFrameNode]:
        return (self.start, self.end)

    @property
    def order_ambiguous(self) -> bool:
        return False

    @property
    def explicitly_ordered(self) -> bool:
        return True

    @functools.cached_property
    def fields(self) -> Iterable[Field]:
        return (Field(self.output_id, next(iter(self.start.fields)).dtype),)

    @functools.cached_property
    def variables_introduced(self) -> int:
        """Defines the number of variables generated by the current node. Used to estimate query planning complexity."""
        return len(self.schema.items) + OVERHEAD_VARIABLES

    @property
    def row_count(self) -> Optional[int]:
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.output_id,)

    @property
    def defines_namespace(self) -> bool:
        return True

    def transform_children(
        self, t: Callable[[BigFrameNode], BigFrameNode]
    ) -> FromRangeNode:
        transformed = dataclasses.replace(self, start=t(self.start), end=t(self.end))
        if self == transformed:
            # reusing existing object speeds up eq, and saves a small amount of memory
            return self
        return transformed

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> FromRangeNode:
        return dataclasses.replace(
            self, output_id=mappings.get(self.output_id, self.output_id)
        )

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> FromRangeNode:
        return self


# Input Nodex
# TODO: Most leaf nodes produce fixed column names based on the datasource
# They should support renaming
@dataclasses.dataclass(frozen=True, eq=False)
class LeafNode(BigFrameNode):
    @property
    def roots(self) -> typing.Set[BigFrameNode]:
        return {self}

    @property
    def fast_offsets(self) -> bool:
        return False

    @property
    def fast_ordered_limit(self) -> bool:
        return False

    def transform_children(self, t: Callable[[BigFrameNode], BigFrameNode]) -> LeafNode:
        return self


class ScanItem(typing.NamedTuple):
    id: bfet_ids.ColumnId
    dtype: bigframes.dtypes.Dtype  # Might be multiple logical types for a given physical source type
    source_id: str  # Flexible enough for both local data and bq data

    def with_id(self, id: bfet_ids.ColumnId) -> ScanItem:
        return ScanItem(id, self.dtype, self.source_id)


@dataclasses.dataclass(frozen=True)
class ScanList:
    items: typing.Tuple[ScanItem, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class ReadLocalNode(LeafNode):
    # TODO: Combine feather_bytes, data_schema, n_rows into a LocalDataDef struct
    feather_bytes: bytes
    data_schema: schemata.ArraySchema
    n_rows: int
    # Mapping of local ids to bfet id.
    scan_list: ScanList
    # Offsets are generated only if this is non-null
    offsets_col: Optional[bigframes.core.identifiers.ColumnId] = None
    session: typing.Optional[bigframes.session.Session] = None

    @property
    def fields(self) -> Iterable[Field]:
        fields = (Field(col_id, dtype) for col_id, dtype, _ in self.scan_list.items)
        if self.offsets_col is not None:
            return itertools.chain(
                fields, (Field(self.offsets_col, bigframes.dtypes.INT_DTYPE),)
            )
        return fields

    @property
    def variables_introduced(self) -> int:
        """Defines the number of variables generated by the current node. Used to estimate query planning complexity."""
        return len(self.scan_list.items) + 1

    @property
    def fast_offsets(self) -> bool:
        return True

    @property
    def fast_ordered_limit(self) -> bool:
        return True

    @property
    def order_ambiguous(self) -> bool:
        return False

    @property
    def explicitly_ordered(self) -> bool:
        return True

    @property
    def row_count(self) -> typing.Optional[int]:
        return self.n_rows

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return tuple(item.id for item in self.fields)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReadLocalNode:
        new_scan_list = ScanList(
            tuple(
                ScanItem(mappings.get(item.id, item.id), item.dtype, item.source_id)
                for item in self.scan_list.items
            )
        )
        new_offsets_col = (
            mappings.get(self.offsets_col, self.offsets_col)
            if (self.offsets_col is not None)
            else None
        )
        return dataclasses.replace(
            self, scan_list=new_scan_list, offsets_col=new_offsets_col
        )

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReadLocalNode:
        return self


@dataclasses.dataclass(frozen=True)
class GbqTable:
    project_id: str = dataclasses.field()
    dataset_id: str = dataclasses.field()
    table_id: str = dataclasses.field()
    physical_schema: Tuple[bq.SchemaField, ...] = dataclasses.field()
    n_rows: int = dataclasses.field()
    is_physically_stored: bool = dataclasses.field()
    cluster_cols: typing.Optional[Tuple[str, ...]]

    @staticmethod
    def from_table(table: bq.Table, columns: Sequence[str] = ()) -> GbqTable:
        # Subsetting fields with columns can reduce cost of row-hash default ordering
        if columns:
            schema = tuple(item for item in table.schema if item.name in columns)
        else:
            schema = tuple(table.schema)
        return GbqTable(
            project_id=table.project,
            dataset_id=table.dataset_id,
            table_id=table.table_id,
            physical_schema=schema,
            n_rows=table.num_rows,
            is_physically_stored=(table.table_type in ["TABLE", "MATERIALIZED_VIEW"]),
            cluster_cols=None
            if table.clustering_fields is None
            else tuple(table.clustering_fields),
        )


@dataclasses.dataclass(frozen=True)
class BigqueryDataSource:
    """
    Google BigQuery Data source.

    This should not be modified once defined, as all attributes contribute to the default ordering.
    """

    table: GbqTable
    at_time: typing.Optional[datetime.datetime] = None
    # Added for backwards compatibility, not validated
    sql_predicate: typing.Optional[str] = None
    ordering: typing.Optional[orderings.RowOrdering] = None


## Put ordering in here or just add order_by node above?
@dataclasses.dataclass(frozen=True, eq=False)
class ReadTableNode(LeafNode):
    source: BigqueryDataSource
    # Subset of physical schema column
    # Mapping of table schema ids to bfet id.
    scan_list: ScanList

    table_session: bigframes.session.Session = dataclasses.field()

    def _validate(self):
        # enforce invariants
        physical_names = set(map(lambda i: i.name, self.source.table.physical_schema))
        if not set(scan.source_id for scan in self.scan_list.items).issubset(
            physical_names
        ):
            raise ValueError(
                f"Requested schema {self.scan_list} cannot be derived from table schemal {self.source.table.physical_schema}"
            )

    @property
    def session(self):
        return self.table_session

    @property
    def fields(self) -> Iterable[Field]:
        return (Field(col_id, dtype) for col_id, dtype, _ in self.scan_list.items)

    @property
    def relation_ops_created(self) -> int:
        # Assume worst case, where readgbq actually has baked in analytic operation to generate index
        return 3

    @property
    def fast_offsets(self) -> bool:
        # Fast head is only supported when row offsets are available or data is clustered over ordering key.
        return (self.source.ordering is not None) and self.source.ordering.is_sequential

    @property
    def fast_ordered_limit(self) -> bool:
        if self.source.ordering is None:
            return False
        order_cols = self.source.ordering.all_ordering_columns
        # monotonicity would probably be fine
        if not all(col.scalar_expression.is_identity for col in order_cols):
            return False
        order_col_ids = tuple(
            cast(ex.DerefOp, col.scalar_expression).id.name for col in order_cols
        )
        cluster_col_ids = self.source.table.cluster_cols
        if cluster_col_ids is None:
            return False

        return order_col_ids == cluster_col_ids[: len(order_col_ids)]

    @property
    def order_ambiguous(self) -> bool:
        return (
            self.source.ordering is None
        ) or not self.source.ordering.is_total_ordering

    @property
    def explicitly_ordered(self) -> bool:
        return self.source.ordering is not None

    @functools.cached_property
    def variables_introduced(self) -> int:
        return len(self.scan_list.items) + 1

    @property
    def row_count(self) -> typing.Optional[int]:
        if self.source.sql_predicate is None and self.source.table.is_physically_stored:
            return self.source.table.n_rows
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return tuple(item.id for item in self.scan_list.items)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReadTableNode:
        new_scan_list = ScanList(
            tuple(
                ScanItem(mappings.get(item.id, item.id), item.dtype, item.source_id)
                for item in self.scan_list.items
            )
        )
        return dataclasses.replace(self, scan_list=new_scan_list)

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReadTableNode:
        return self

    def with_order_cols(self):
        # Maybe the ordering should be required to always be in the scan list, and then we won't need this?
        if self.source.ordering is None:
            return self, orderings.RowOrdering()

        order_cols = {col.sql for col in self.source.ordering.referenced_columns}
        scan_cols = {col.source_id for col in self.scan_list.items}
        new_scan_cols = [
            ScanItem(
                bigframes.core.ids.ColumnId.unique(),
                dtype=bigframes.dtypes.convert_schema_field(field)[1],
                source_id=field.name,
            )
            for field in self.source.table.physical_schema
            if (field.name in order_cols) and (field.name not in scan_cols)
        ]
        new_scan_list = ScanList(items=(*self.scan_list.items, *new_scan_cols))
        new_order = self.source.ordering.remap_column_refs(
            {
                bigframes.core.ids.ColumnId(item.source_id): item.id
                for item in new_scan_cols
            },
            allow_partial_bindings=True,
        )
        return dataclasses.replace(self, scan_list=new_scan_list), new_order


@dataclasses.dataclass(frozen=True, eq=False)
class CachedTableNode(ReadTableNode):
    # The original BFET subtree that was cached
    # note: this isn't a "child" node.
    original_node: BigFrameNode = dataclasses.field()


# Unary nodes
@dataclasses.dataclass(frozen=True, eq=False)
class PromoteOffsetsNode(UnaryNode, AdditiveNode):
    col_id: bigframes.core.identifiers.ColumnId

    @property
    def non_local(self) -> bool:
        return True

    @property
    def fields(self) -> Iterable[Field]:
        return itertools.chain(
            self.child.fields, [Field(self.col_id, bigframes.dtypes.INT_DTYPE)]
        )

    @property
    def relation_ops_created(self) -> int:
        return 2

    @functools.cached_property
    def variables_introduced(self) -> int:
        return 1

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.col_id,)

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset()

    @property
    def added_fields(self) -> Tuple[Field, ...]:
        return (Field(self.col_id, bigframes.dtypes.INT_DTYPE),)

    @property
    def additive_base(self) -> BigFrameNode:
        return self.child

    def replace_additive_base(self, node: BigFrameNode) -> PromoteOffsetsNode:
        return dataclasses.replace(self, child=node)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> PromoteOffsetsNode:
        return dataclasses.replace(self, col_id=mappings.get(self.col_id, self.col_id))

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> PromoteOffsetsNode:
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class FilterNode(UnaryNode):
    predicate: ex.Expression

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def variables_introduced(self) -> int:
        return 1

    @property
    def row_count(self) -> Optional[int]:
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(self.ids) | self.referenced_ids

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset(self.predicate.column_references)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> FilterNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> FilterNode:
        return dataclasses.replace(
            self,
            predicate=self.predicate.remap_column_refs(
                mappings, allow_partial_bindings=True
            ),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class OrderByNode(UnaryNode):
    by: Tuple[OrderingExpression, ...]
    # This is an optimization, if true, can discard previous orderings.
    # might be a total ordering even if false
    is_total_order: bool = False

    @property
    def variables_introduced(self) -> int:
        return 0

    @property
    def relation_ops_created(self) -> int:
        # Doesnt directly create any relational operations
        return 0

    @property
    def explicitly_ordered(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(self.ids) | self.referenced_ids

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset(
            itertools.chain.from_iterable(map(lambda x: x.referenced_columns, self.by))
        )

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> OrderByNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> OrderByNode:
        all_refs = set(
            itertools.chain.from_iterable(map(lambda x: x.referenced_columns, self.by))
        )
        ref_mapping = {id: ex.DerefOp(mappings[id]) for id in all_refs}
        new_by = cast(
            tuple[OrderingExpression, ...],
            tuple(
                by_expr.bind_refs(ref_mapping, allow_partial_bindings=True)
                for by_expr in self.by
            ),
        )
        return dataclasses.replace(self, by=new_by)


@dataclasses.dataclass(frozen=True, eq=False)
class ReversedNode(UnaryNode):
    # useless field to make sure has distinct hash
    reversed: bool = True

    @property
    def variables_introduced(self) -> int:
        return 0

    @property
    def relation_ops_created(self) -> int:
        # Doesnt directly create any relational operations
        return 0

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset()

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReversedNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ReversedNode:
        return self


class AliasedRef(typing.NamedTuple):
    ref: ex.DerefOp
    id: bfet_ids.ColumnId

    @classmethod
    def identity(cls, id: bfet_ids.ColumnId) -> AliasedRef:
        return cls(ex.DerefOp(id), id)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> AliasedRef:
        return AliasedRef(self.ref, mappings.get(self.id, self.id))

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> AliasedRef:
        return AliasedRef(ex.DerefOp(mappings.get(self.ref.id, self.ref.id)), self.id)


@dataclasses.dataclass(frozen=True, eq=False)
class SelectionNode(UnaryNode):
    input_output_pairs: Tuple[AliasedRef, ...]

    def _validate(self):
        for ref, _ in self.input_output_pairs:
            if ref.id not in set(self.child.ids):
                raise ValueError(f"Reference to column not in child: {ref.id}")

    @functools.cached_property
    def fields(self) -> Iterable[Field]:
        return tuple(
            Field(output, self.child.get_type(ref.id))
            for ref, output in self.input_output_pairs
        )

    @property
    def variables_introduced(self) -> int:
        # This operation only renames variables, doesn't actually create new ones
        return 0

    # TODO: Reuse parent namespace
    # Currently, Selection node allows renaming an reusing existing names, so it must establish a
    # new namespace.
    @property
    def defines_namespace(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return tuple(id for _, id in self.input_output_pairs)

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(ref.id for ref, id in self.input_output_pairs)

    def get_id_mapping(self) -> dict[bfet_ids.ColumnId, bfet_ids.ColumnId]:
        return {ref.id: id for ref, id in self.input_output_pairs}

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> SelectionNode:
        new_fields = tuple(
            item.remap_vars(mappings) for item in self.input_output_pairs
        )
        return dataclasses.replace(self, input_output_pairs=new_fields)

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> SelectionNode:
        new_fields = tuple(
            item.remap_refs(mappings) for item in self.input_output_pairs
        )
        return dataclasses.replace(self, input_output_pairs=new_fields)  # type: ignore


@dataclasses.dataclass(frozen=True, eq=False)
class ProjectionNode(UnaryNode, AdditiveNode):
    """Assigns new variables (without modifying existing ones)"""

    assignments: typing.Tuple[
        typing.Tuple[ex.Expression, bigframes.core.identifiers.ColumnId], ...
    ]

    def _validate(self):
        input_types = self.child._dtype_lookup
        for expression, id in self.assignments:
            # throws TypeError if invalid
            _ = expression.output_type(input_types)
        # Cannot assign to existing variables - append only!
        assert all(name not in self.child.schema.names for _, name in self.assignments)

    @functools.cached_property
    def added_fields(self) -> Tuple[Field, ...]:
        input_types = self.child._dtype_lookup
        return tuple(
            Field(id, bigframes.dtypes.dtype_for_etype(ex.output_type(input_types)))
            for ex, id in self.assignments
        )

    @property
    def fields(self) -> Iterable[Field]:
        return itertools.chain(self.child.fields, self.added_fields)

    @property
    def variables_introduced(self) -> int:
        # ignore passthrough expressions
        new_vars = sum(1 for i in self.assignments if not i[0].is_identity)
        return new_vars

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return tuple(id for _, id in self.assignments)

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(
            itertools.chain.from_iterable(
                i[0].column_references for i in self.assignments
            )
        )

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset(
            itertools.chain.from_iterable(
                ex.column_references for ex, id in self.assignments
            )
        )

    @property
    def additive_base(self) -> BigFrameNode:
        return self.child

    def replace_additive_base(self, node: BigFrameNode) -> ProjectionNode:
        return dataclasses.replace(self, child=node)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ProjectionNode:
        new_fields = tuple((ex, mappings.get(id, id)) for ex, id in self.assignments)
        return dataclasses.replace(self, assignments=new_fields)

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ProjectionNode:
        new_fields = tuple(
            (ex.remap_column_refs(mappings, allow_partial_bindings=True), id)
            for ex, id in self.assignments
        )
        return dataclasses.replace(self, assignments=new_fields)


# TODO: Merge RowCount into Aggregate Node?
# Row count can be compute from table metadata sometimes, so it is a bit special.
@dataclasses.dataclass(frozen=True, eq=False)
class RowCountNode(UnaryNode):
    col_id: bfet_ids.ColumnId = bfet_ids.ColumnId("count")

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def non_local(self) -> bool:
        return True

    @property
    def fields(self) -> Iterable[Field]:
        return (Field(self.col_id, bigframes.dtypes.INT_DTYPE),)

    @property
    def variables_introduced(self) -> int:
        return 1

    @property
    def defines_namespace(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        return 1

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.col_id,)

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset()

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> RowCountNode:
        return dataclasses.replace(self, col_id=mappings.get(self.col_id, self.col_id))

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> RowCountNode:
        return self


@dataclasses.dataclass(frozen=True, eq=False)
class AggregateNode(UnaryNode):
    aggregations: typing.Tuple[
        typing.Tuple[ex.Aggregation, bigframes.core.identifiers.ColumnId], ...
    ]
    by_column_ids: typing.Tuple[ex.DerefOp, ...] = tuple([])
    order_by: Tuple[OrderingExpression, ...] = ()
    dropna: bool = True

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def non_local(self) -> bool:
        return True

    @functools.cached_property
    def fields(self) -> Iterable[Field]:
        by_items = (
            Field(ref.id, self.child.get_type(ref.id)) for ref in self.by_column_ids
        )
        agg_items = (
            Field(
                id,
                bigframes.dtypes.dtype_for_etype(
                    agg.output_type(self.child._dtype_lookup)
                ),
            )
            for agg, id in self.aggregations
        )
        return tuple(itertools.chain(by_items, agg_items))

    @property
    def variables_introduced(self) -> int:
        return len(self.aggregations) + len(self.by_column_ids)

    @property
    def order_ambiguous(self) -> bool:
        return False

    @property
    def explicitly_ordered(self) -> bool:
        return True

    @property
    def row_count(self) -> Optional[int]:
        if not self.by_column_ids:
            return 1
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return tuple(id for _, id in self.aggregations)

    @property
    def consumed_ids(self) -> COLUMN_SET:
        by_ids = (ref.id for ref in self.by_column_ids)
        agg_inputs = itertools.chain.from_iterable(
            agg.column_references for agg, _ in self.aggregations
        )
        order_ids = itertools.chain.from_iterable(
            part.scalar_expression.column_references for part in self.order_by
        )
        return frozenset(itertools.chain(by_ids, agg_inputs, order_ids))

    @property
    def has_ordered_ops(self) -> bool:
        return not all(
            aggregate.op.order_independent for aggregate, _ in self.aggregations
        )

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> AggregateNode:
        new_aggs = tuple((agg, mappings.get(id, id)) for agg, id in self.aggregations)
        return dataclasses.replace(self, aggregations=new_aggs)

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> AggregateNode:
        new_aggs = tuple(
            (agg.remap_column_refs(mappings, allow_partial_bindings=True), id)
            for agg, id in self.aggregations
        )
        new_by_ids = tuple(id.remap_column_refs(mappings) for id in self.by_column_ids)
        new_order_by = tuple(part.remap_column_refs(mappings) for part in self.order_by)
        return dataclasses.replace(
            self, by_column_ids=new_by_ids, aggregations=new_aggs, order_by=new_order_by
        )


@dataclasses.dataclass(frozen=True, eq=False)
class WindowOpNode(UnaryNode, AdditiveNode):
    expression: ex.Aggregation
    window_spec: window.WindowSpec
    output_name: bigframes.core.identifiers.ColumnId
    never_skip_nulls: bool = False
    skip_reproject_unsafe: bool = False

    def _validate(self):
        """Validate the local data in the node."""
        # Since inner order and row bounds are coupled, rank ops can't be row bounded
        assert (
            not self.window_spec.row_bounded
        ) or self.expression.op.implicitly_inherits_order
        assert all(ref in self.child.ids for ref in self.expression.column_references)

    @property
    def non_local(self) -> bool:
        return True

    @property
    def fields(self) -> Iterable[Field]:
        return itertools.chain(self.child.fields, [self.added_field])

    @property
    def variables_introduced(self) -> int:
        return 1

    @property
    def added_fields(self) -> Tuple[Field, ...]:
        return (self.added_field,)

    @property
    def relation_ops_created(self) -> int:
        # Assume that if not reprojecting, that there is a sequence of window operations sharing the same window
        return 0 if self.skip_reproject_unsafe else 4

    @property
    def row_count(self) -> Optional[int]:
        return self.child.row_count

    @functools.cached_property
    def added_field(self) -> Field:
        input_types = self.child._dtype_lookup
        return Field(
            self.output_name,
            bigframes.dtypes.dtype_for_etype(self.expression.output_type(input_types)),
        )

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.output_name,)

    @property
    def consumed_ids(self) -> COLUMN_SET:
        return frozenset(
            set(self.ids).difference([self.output_name]).union(self.referenced_ids)
        )

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return (
            frozenset()
            .union(self.expression.column_references)
            .union(self.window_spec.all_referenced_columns)
        )

    @property
    def inherits_order(self) -> bool:
        # does the op both use ordering at all? and if so, can it inherit order?
        op_inherits_order = (
            not self.expression.op.order_independent
        ) and self.expression.op.implicitly_inherits_order
        return op_inherits_order or self.window_spec.row_bounded

    @property
    def additive_base(self) -> BigFrameNode:
        return self.child

    def replace_additive_base(self, node: BigFrameNode) -> WindowOpNode:
        return dataclasses.replace(self, child=node)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> WindowOpNode:
        return dataclasses.replace(
            self, output_name=mappings.get(self.output_name, self.output_name)
        )

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> WindowOpNode:
        return dataclasses.replace(
            self,
            expression=self.expression.remap_column_refs(
                mappings, allow_partial_bindings=True
            ),
            window_spec=self.window_spec.remap_column_refs(
                mappings, allow_partial_bindings=True
            ),
        )


@dataclasses.dataclass(frozen=True, eq=False)
class RandomSampleNode(UnaryNode):
    fraction: float

    @property
    def deterministic(self) -> bool:
        return False

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def variables_introduced(self) -> int:
        return 1

    @property
    def row_count(self) -> Optional[int]:
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return ()

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset()

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> RandomSampleNode:
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> RandomSampleNode:
        return self


# TODO: Explode should create a new column instead of overriding the existing one
@dataclasses.dataclass(frozen=True, eq=False)
class ExplodeNode(UnaryNode):
    column_ids: typing.Tuple[ex.DerefOp, ...]
    # Offsets are generated only if this is non-null
    offsets_col: Optional[bigframes.core.identifiers.ColumnId] = None

    @property
    def row_preserving(self) -> bool:
        return False

    @property
    def fields(self) -> Iterable[Field]:
        fields = (
            Field(
                field.id,
                bigframes.dtypes.arrow_dtype_to_bigframes_dtype(
                    self.child.get_type(field.id).pyarrow_dtype.value_type  # type: ignore
                ),
            )
            if field.id in set(map(lambda x: x.id, self.column_ids))
            else field
            for field in self.child.fields
        )
        if self.offsets_col is not None:
            return itertools.chain(
                fields, (Field(self.offsets_col, bigframes.dtypes.INT_DTYPE),)
            )
        return fields

    @property
    def relation_ops_created(self) -> int:
        return 3

    @functools.cached_property
    def variables_introduced(self) -> int:
        return len(self.column_ids) + 1

    @property
    def row_count(self) -> Optional[int]:
        return None

    @property
    def node_defined_ids(self) -> Tuple[bfet_ids.ColumnId, ...]:
        return (self.offsets_col,) if (self.offsets_col is not None) else ()

    @property
    def referenced_ids(self) -> COLUMN_SET:
        return frozenset(ref.id for ref in self.column_ids)

    def remap_vars(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ExplodeNode:
        if (self.offsets_col is not None) and self.offsets_col in mappings:
            return dataclasses.replace(self, offsets_col=mappings[self.offsets_col])
        return self

    def remap_refs(
        self, mappings: Mapping[bfet_ids.ColumnId, bfet_ids.ColumnId]
    ) -> ExplodeNode:
        new_ids = tuple(id.remap_column_refs(mappings) for id in self.column_ids)
        return dataclasses.replace(self, column_ids=new_ids)  # type: ignore


# Tree operators
def top_down(
    root: BigFrameNode,
    transform: Callable[[BigFrameNode], BigFrameNode],
    *,
    memoize=False,
    validate=False,
) -> BigFrameNode:
    """
    Perform a top-down transformation of the BigFrameNode tree.

    If memoize=True, recursive calls are memoized within the scope of the traversal only.
    """

    def top_down_internal(root: BigFrameNode) -> BigFrameNode:
        return transform(root).transform_children(top_down_internal)

    if memoize:
        # MUST reassign to the same name or caching won't work recursively
        top_down_internal = functools.cache(top_down_internal)

    result = top_down_internal(root)
    if validate:
        result.validate_tree()
    return result


def bottom_up(
    root: BigFrameNode,
    transform: Callable[[BigFrameNode], BigFrameNode],
    *,
    memoize=False,
    validate=False,
) -> BigFrameNode:
    """
    Perform a bottom-up transformation of the BigFrameNode tree.

    If memoize=True, recursive calls are memoized within the scope of the traversal only.
    """

    def bottom_up_internal(root: BigFrameNode) -> BigFrameNode:
        return transform(root.transform_children(bottom_up_internal))

    if memoize:
        # MUST reassign to the same name or caching won't work recursively
        bottom_up_internal = functools.cache(bottom_up_internal)

    result = bottom_up_internal(root)
    if validate:
        result.validate_tree()
    return result
