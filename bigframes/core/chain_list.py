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

import collections.abc
import functools
from typing import Sequence, TypeVar

ColumnIdentifierType = str


T = TypeVar("T")

# Further optimizations possible:
# * Support mapping operators
# * Support insertions and deletions


class ChainList(collections.abc.Sequence[T]):
    def __init__(self, *parts: Sequence[T]):
        # Could maybe decompose child chainlists?
        self._parts = tuple(parts)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        offset = 0
        # Could do binary search
        for part in self._parts:
            if (index - offset) < len(part):
                return part[index - offset]
            offset += len(part)
        raise IndexError("Index out of bounds")

    @functools.cache
    def __len__(self):
        return sum(map(len, self._parts))

    def __iter__(self):
        for part in self._parts:
            yield from part
