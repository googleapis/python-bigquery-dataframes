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

import typing

import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.operations as ops
import bigframes.operations.aggregations as agg_ops


def indicate_duplicates(
    block: blocks.Block, columns: typing.Sequence[str], keep: str = "first"
) -> typing.Tuple[blocks.Block, str]:
    """Create a boolean column where True indicates a duplicate value"""
    if keep not in ["first", "last", False]:
        raise ValueError("keep must be one of 'first', 'last', or False'")

    if keep == "first":
        # Count how many copies occur up to current copy of value
        # Discard this value if there are copies BEFORE
        window_spec = core.WindowSpec(
            grouping_keys=tuple(columns),
            following=0,
        )
    elif keep == "last":
        # Count how many copies occur up to current copy of values
        # Discard this value if there are copies AFTER
        window_spec = core.WindowSpec(
            grouping_keys=tuple(columns),
            preceding=0,
        )
    else:  # keep == False
        # Count how many copies of the value occur in entire series.
        # Discard this value if there are copies ANYWHERE
        window_spec = core.WindowSpec(grouping_keys=tuple(columns))
    block, dummy = block.create_constant(1)
    block, val_count_col_id = block.apply_window_op(
        dummy,
        agg_ops.count_op,
        window_spec=window_spec,
    )
    block, duplicate_indicator = block.apply_unary_op(
        val_count_col_id,
        ops.partial_right(ops.gt_op, 1),
    )
    return (
        block.drop_columns(
            (
                dummy,
                val_count_col_id,
            )
        ),
        duplicate_indicator,
    )


def drop_duplicates(
    block: blocks.Block, columns: typing.Sequence[str], keep: str = "first"
) -> blocks.Block:
    block, dupe_indicator_id = indicate_duplicates(block, columns, keep)
    block, keep_indicator_id = block.apply_unary_op(dupe_indicator_id, ops.invert_op)
    return block.filter(keep_indicator_id).drop_columns(
        (dupe_indicator_id, keep_indicator_id)
    )
