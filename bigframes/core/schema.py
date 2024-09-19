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

from __future__ import annotations

from dataclasses import dataclass
import functools
import typing

import google.cloud.bigquery

import bigframes.core.guid
import bigframes.core.identifiers
import bigframes.dtypes


@dataclass(frozen=True)
class SchemaItem:
    column: bigframes.core.identifiers.Identifier
    dtype: bigframes.dtypes.Dtype


@dataclass(frozen=True)
class ArraySchema:
    items: typing.Tuple[SchemaItem, ...]

    @classmethod
    def from_bq_table(cls, table: google.cloud.bigquery.Table):
        items = tuple(
            SchemaItem(bigframes.core.identifiers.SimpleIdentifier(name), dtype)
            for name, dtype in bigframes.dtypes.bf_type_from_type_kind(
                table.schema
            ).items()
        )
        return ArraySchema(items)

    @property
    def ids(self) -> typing.Tuple[bigframes.core.identifiers.Identifier, ...]:
        return tuple(item.column for item in self.items)

    @property
    def names(self) -> typing.Tuple[str, ...]:
        return tuple(item.column.name for item in self.items)

    @property
    def dtypes(self) -> typing.Tuple[bigframes.dtypes.Dtype, ...]:
        return tuple(item.dtype for item in self.items)

    @functools.cached_property
    def _name_mapping(
        self,
    ) -> typing.Dict[str, SchemaItem]:
        if len(self.names) > len(set(self.names)):
            # ArrayValue schemas will always be unambiguous, but after rewrites, local name references
            # should not be used, instead, resolve references to plan-unique id.
            raise ValueError(
                "Schema names are non-unique, columns cannot be referenced by name"
            )
        return {item.column.name: item for item in self.items}

    def to_bigquery(self) -> typing.Tuple[google.cloud.bigquery.SchemaField, ...]:
        return tuple(
            bigframes.dtypes.convert_to_schema_field(item.column.name, item.dtype)
            for item in self.items
        )

    def select(self, columns: typing.Iterable[str]) -> ArraySchema:
        return ArraySchema(
            tuple(
                self.resolve_ref(bigframes.core.identifiers.NameReference(name))
                for name in columns
            )
        )

    def append(self, item: SchemaItem):
        return ArraySchema(tuple([*self.items, item]))

    def prepend(self, item: SchemaItem):
        return ArraySchema(tuple([item, *self.items]))

    def get_type(self, ref: bigframes.core.identifiers.ColumnReference):
        return self.resolve_ref(ref).dtype

    def resolve_ref(
        self, ref: bigframes.core.identifiers.ColumnReference
    ) -> SchemaItem:
        if isinstance(ref, bigframes.core.identifiers.OffsetReference):
            return self.items[ref.offset]
        if isinstance(ref, bigframes.core.identifiers.NameReference):
            return self._name_mapping[ref.name]
        if isinstance(ref, bigframes.core.identifiers.IdReference):
            raise ValueError("Id references not yet supported")
        raise ValueError(f"Unrecognized column reference: {ref}")

    def __len__(self) -> int:
        return len(self.items)
