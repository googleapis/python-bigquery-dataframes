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

# Later, plan on migrating ids to use integers to reduce memory usage allow use of bitmaps to represent column sets
from __future__ import annotations

import abc
import dataclasses
from typing import Generator


def simple(name: str) -> SimpleIdentifier:
    return SimpleIdentifier(name)


def standard_identifiers() -> Generator[str, None, None]:
    i = 0
    while True:
        yield f"col_{i}"
        i = i + 1


# Identifiers are used in three different contexts that we want to be able to move between
# 1. ArrayValue interface. At this level, identifiers are used to refer to unambiguously reference a column.
# 2. Implicit joiner. Here, nodes from two trees are merged.
# 3. Plan optimization. Here, global addressing is useful so that references don't need to be remapped when nodes are reordered
# 4. SQL generation. Like at array value, local uniqueness is what matters. These must be strings
class Identifier(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The canonical name for the identifier. Probably not globally unique."""
        ...

    @property
    @abc.abstractmethod
    def ref(self) -> ColumnReference:
        """Generate the most robust column reference possible with current information."""
        ...


@dataclasses.dataclass(frozen=True)
class SimpleIdentifier(Identifier):
    """An identifier with a locally unambiguous name, but no other metadata."""

    name: str

    @property
    def ref(self) -> ColumnReference:
        return NameReference(self.name)


# References are used to refer to variables defined elsewhere
# Some reference types may be contextual, while other may not be.


class ColumnReference(abc.ABC):
    ...


@dataclasses.dataclass(frozen=True)
class NameReference(ColumnReference):
    """Contextual reference that refers to a column name in the input schema."""

    name: str


@dataclasses.dataclass(frozen=True)
class OffsetReference(ColumnReference):
    offset: int


@dataclasses.dataclass(frozen=True)
class IdReference(ColumnReference):
    id: int
