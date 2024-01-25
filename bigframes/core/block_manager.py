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
from typing import Optional

import bigframes.core.blocks as blocks


@dataclasses.dataclass(frozen=True)
class ViewDef:
    slice_start: int = 0
    slice_stop: Optional[int] = None

    def apply(self, block: blocks.Block) -> blocks.Block:
        return block.slice(self.slice_start, self.slice_stop)

    def chain(self, view_def: ViewDef) -> ViewDef:
        new_start = self.slice_start + view_def.slice_start
        new_stop: Optional[int] = None
        if (view_def.slice_stop is not None) and (self.slice_stop is not None):
            new_stop = min(self.slice_start + view_def.slice_stop, self.slice_stop)
        elif view_def.slice_stop is not None:
            new_stop = self.slice_start + view_def.slice_stop
        else:
            new_stop = self.slice_stop
        return ViewDef(slice_start=new_start, slice_stop=new_stop)


def create_block_provider(block: blocks.Block) -> BlockProvider:
    return DirectBlockProvider(block)


class BlockProvider(abc.ABC):
    """A block provider provides a layer between Block consumers (DataFrame/Series/Index) and the specific block implementation.
    This provides freedom to modify block internal representations to optimize execution, without modifying semantics."""

    @abc.abstractmethod
    def cache(self):
        """Materialize the underlying block and rewrite to reference the materialized table."""
        ...

    @abc.abstractmethod
    def create_view(self, view_def: ViewDef) -> BlockViewProvider:
        """Crate a view of the provided block."""
        ...

    @abc.abstractmethod
    def to_local(self, options: blocks.MaterializationOptions):
        """Materialize the data to a local pandas dataframe."""
        ...

    @abc.abstractmethod
    def get_block(self) -> blocks.Block:
        """Get the underlying block value."""
        ...


@dataclasses.dataclass
class DirectBlockProvider(BlockProvider):
    _block: blocks.Block

    def cache(self, optimize_offsets: bool = False):
        self._block = self._block.cached(optimize_offsets=optimize_offsets)

    def create_view(self, view_def: ViewDef) -> BlockViewProvider:
        return BlockViewProvider(self, view_def)

    def to_local(self, options: blocks.MaterializationOptions):
        return self._block.to_pandas(options)

    def get_block(self) -> blocks.Block:
        return self._block


@dataclasses.dataclass
class BlockViewProvider(BlockProvider):
    _base: DirectBlockProvider
    _view: ViewDef

    def create_view(self, view_def: ViewDef) -> BlockViewProvider:
        return BlockViewProvider(self._base, self._view.chain(view_def=view_def))

    def cache(self, optimize_offsets: bool = False):
        self._base.cache(optimize_offsets=True)

    def to_local(self, options: blocks.MaterializationOptions):
        self._base.cache(optimize_offsets=True)
        view_block = self._view.apply(self._base.get_block())
        return view_block.to_pandas(options)

    def get_block(self) -> blocks.Block:
        return self._view.apply(self._base.get_block())
