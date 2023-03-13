from __future__ import annotations

import functools
import typing

import bigframes.core.blocks as blocks


class ViewWindow:
    """Base class for view windows."""

    def apply_window(self, block: blocks.Block) -> blocks.Block:
        return block


class SliceViewWindow:
    """View window representing a slice operator with a start, stop and step component."""

    def __init__(
        self,
        start: typing.Optional[int] = None,
        stop: typing.Optional[int] = None,
        step: typing.Optional[int] = None,
    ):
        self._start = start
        self._stop = stop
        self._step = step

    def apply_window(self, block: blocks.Block) -> blocks.Block:
        expr_with_offsets = block.expr.project_offsets()
        cond_list = []
        # TODO(tbergeron): Handle negative indexing
        if self._start:
            cond_list.append(expr_with_offsets.offsets >= self._start)
        if self._stop:
            cond_list.append(expr_with_offsets.offsets < self._stop)
        if self._step:
            # TODO(tbergeron): Reverse the ordering if negative step
            start = self._start if self._start else 0
            cond_list.append((expr_with_offsets.offsets - start) % self._step == 0)
        if not cond_list:
            return block
        return blocks.Block(
            expr_with_offsets.filter(functools.reduce(lambda x, y: x & y, cond_list)),
            index_columns=block.index_columns,
        )
