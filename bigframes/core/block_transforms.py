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

import pandas as pd

import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.ordering as ordering
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


def value_counts(
    block: blocks.Block,
    columns: typing.Sequence[str],
    normalize: bool = False,
    sort: bool = True,
    ascending: bool = False,
    dropna: bool = True,
):
    block, dummy = block.create_constant(1)
    block, agg_ids = block.aggregate(
        by_column_ids=columns,
        aggregations=[(dummy, agg_ops.count_op)],
        dropna=dropna,
        as_index=True,
    )
    count_id = agg_ids[0]
    if normalize:
        unbound_window = core.WindowSpec()
        block, total_count_id = block.apply_window_op(
            count_id, agg_ops.sum_op, unbound_window
        )
        block, count_id = block.apply_binary_op(count_id, total_count_id, ops.div_op)

    if sort:
        block = block.order_by(
            [
                ordering.OrderingColumnReference(
                    count_id,
                    direction=ordering.OrderingDirection.ASC
                    if ascending
                    else ordering.OrderingDirection.DESC,
                )
            ]
        )
    return block.select_column(count_id).with_column_labels(["count"])


def pct_change(block: blocks.Block, periods: int = 1) -> blocks.Block:
    column_labels = block.column_labels
    window_spec = core.WindowSpec(
        preceding=periods if periods > 0 else None,
        following=-periods if periods < 0 else None,
    )

    original_columns = block.value_columns
    block, shift_columns = block.multi_apply_window_op(
        original_columns, agg_ops.ShiftOp(periods), window_spec=window_spec
    )
    result_ids = []
    for original_col, shifted_col in zip(original_columns, shift_columns):
        block, change_id = block.apply_binary_op(original_col, shifted_col, ops.sub_op)
        block, pct_change_id = block.apply_binary_op(change_id, shifted_col, ops.div_op)
        result_ids.append(pct_change_id)
    return block.select_columns(result_ids).with_column_labels(column_labels)


def rank(
    block: blocks.Block,
    method: str = "average",
    na_option: str = "keep",
    ascending: bool = True,
):
    if method not in ["average", "min", "max", "first", "dense"]:
        raise ValueError(
            "method must be one of 'average', 'min', 'max', 'first', or 'dense'"
        )
    if na_option not in ["keep", "top", "bottom"]:
        raise ValueError("na_option must be one of 'keep', 'top', or 'bottom'")

    columns = block.value_columns
    labels = block.column_labels
    # Step 1: Calculate row numbers for each row
    # Identify null values to be treated according to na_option param
    rownum_col_ids = []
    nullity_col_ids = []
    for col in columns:
        block, nullity_col_id = block.apply_unary_op(
            col,
            ops.isnull_op,
        )
        nullity_col_ids.append(nullity_col_id)
        window = core.WindowSpec(
            # BigQuery has syntax to reorder nulls with "NULLS FIRST/LAST", but that is unavailable through ibis presently, so must order on a separate nullity expression first.
            ordering=(
                ordering.OrderingColumnReference(
                    col,
                    ordering.OrderingDirection.ASC
                    if ascending
                    else ordering.OrderingDirection.DESC,
                    na_last=(na_option in ["bottom", "keep"]),
                ),
            ),
        )
        # Count_op ignores nulls, so if na_option is "top" or "bottom", we instead count the nullity columns, where nulls have been mapped to bools
        block, rownum_id = block.apply_window_op(
            col if na_option == "keep" else nullity_col_id,
            agg_ops.dense_rank_op if method == "dense" else agg_ops.count_op,
            window_spec=window,
            skip_reproject_unsafe=(col != columns[-1]),
        )
        rownum_col_ids.append(rownum_id)

    # Step 2: Apply aggregate to groups of like input values.
    # This step is skipped for method=='first' or 'dense'
    if method in ["average", "min", "max"]:
        agg_op = {
            "average": agg_ops.mean_op,
            "min": agg_ops.min_op,
            "max": agg_ops.max_op,
        }[method]
        post_agg_rownum_col_ids = []
        for i in range(len(columns)):
            block, result_id = block.apply_window_op(
                rownum_col_ids[i],
                agg_op,
                window_spec=core.WindowSpec(grouping_keys=[columns[i]]),
                skip_reproject_unsafe=(i < (len(columns) - 1)),
            )
            post_agg_rownum_col_ids.append(result_id)
        rownum_col_ids = post_agg_rownum_col_ids

    # Step 3: post processing: mask null values and cast to float
    if method in ["min", "max", "first", "dense"]:
        # Pandas rank always produces Float64, so must cast for aggregation types that produce ints
        block = block.multi_apply_unary_op(
            rownum_col_ids, ops.AsTypeOp(pd.Float64Dtype())
        )
    if na_option == "keep":
        # For na_option "keep", null inputs must produce null outputs
        for i in range(len(columns)):
            block, null_const = block.create_constant(pd.NA, dtype=pd.Float64Dtype())
            block, rownum_col_ids[i] = block.apply_ternary_op(
                null_const, nullity_col_ids[i], rownum_col_ids[i], ops.where_op
            )

    return block.select_columns(rownum_col_ids).with_column_labels(labels)


def dropna(block: blocks.Block, how: typing.Literal["all", "any"] = "any"):
    """
    Drop na entries from block
    """
    if how == "any":
        filtered_block = block
        for column in block.value_columns:
            filtered_block, result_id = filtered_block.apply_unary_op(
                column, ops.notnull_op
            )
            filtered_block = filtered_block.filter(result_id)
            filtered_block = filtered_block.drop_columns([result_id])
        return filtered_block
    else:  # "all"
        filtered_block = block
        predicate = None
        for column in block.value_columns:
            filtered_block, partial_predicate = filtered_block.apply_unary_op(
                column, ops.notnull_op
            )
            if predicate:
                filtered_block, predicate = filtered_block.apply_binary_op(
                    partial_predicate, predicate, ops.or_op
                )
            else:
                predicate = partial_predicate
        if predicate:
            filtered_block = filtered_block.filter(predicate)
        filtered_block = filtered_block.select_columns(block.value_columns)
        return filtered_block


def nsmallest(
    block: blocks.Block,
    n: int,
    column_ids: typing.Sequence[str],
    keep: str,
) -> blocks.Block:
    if keep not in ("first", "last", "all"):
        raise ValueError("'keep must be one of 'first', 'last', or 'all'")
    if keep == "last":
        block = block.reversed()
    order_refs = [
        ordering.OrderingColumnReference(
            col_id, direction=ordering.OrderingDirection.ASC
        )
        for col_id in column_ids
    ]
    block = block.order_by(order_refs, stable=True)
    if keep in ("first", "last"):
        return block.slice(0, n)
    else:  # keep == "all":
        block, counter = block.apply_window_op(
            column_ids[0],
            agg_ops.rank_op,
            window_spec=core.WindowSpec(ordering=order_refs),
        )
        block, condition = block.apply_unary_op(
            counter, ops.partial_right(ops.le_op, n)
        )
        block = block.filter(condition)
        return block.drop_columns([counter, condition])


def nlargest(
    block: blocks.Block,
    n: int,
    column_ids: typing.Sequence[str],
    keep: str,
) -> blocks.Block:
    if keep not in ("first", "last", "all"):
        raise ValueError("'keep must be one of 'first', 'last', or 'all'")
    if keep == "last":
        block = block.reversed()
    order_refs = [
        ordering.OrderingColumnReference(
            col_id, direction=ordering.OrderingDirection.DESC
        )
        for col_id in column_ids
    ]
    block = block.order_by(order_refs, stable=True)
    if keep in ("first", "last"):
        return block.slice(0, n)
    else:  # keep == "all":
        block, counter = block.apply_window_op(
            column_ids[0],
            agg_ops.rank_op,
            window_spec=core.WindowSpec(ordering=order_refs),
        )
        block, condition = block.apply_unary_op(
            counter, ops.partial_right(ops.le_op, n)
        )
        block = block.filter(condition)
        return block.drop_columns([counter, condition])


def skew(
    block: blocks.Block,
    skew_column_ids: typing.Sequence[str],
    grouping_column_ids: typing.Sequence[str] = (),
) -> blocks.Block:

    original_columns = skew_column_ids
    column_labels = block.select_columns(original_columns).column_labels

    block, delta3_ids = _mean_delta_to_power(
        block, 3, original_columns, grouping_column_ids
    )
    # counts, moment3 for each column
    aggregations = []
    for i, col in enumerate(original_columns):
        count_agg = (col, agg_ops.count_op)
        moment3_agg = (delta3_ids[i], agg_ops.mean_op)
        variance_agg = (col, agg_ops.PopVarOp())
        aggregations.extend([count_agg, moment3_agg, variance_agg])

    block, agg_ids = block.aggregate(
        by_column_ids=grouping_column_ids, aggregations=aggregations
    )

    skew_ids = []
    for i, col in enumerate(original_columns):
        # Corresponds to order of aggregations in preceding loop
        count_id, moment3_id, var_id = agg_ids[i * 3 : (i * 3) + 3]
        block, skew_id = _skew_from_moments_and_count(
            block, count_id, moment3_id, var_id
        )
        skew_ids.append(skew_id)

    block = block.select_columns(skew_ids).with_column_labels(column_labels)
    if not grouping_column_ids:
        # When ungrouped, stack everything into single column so can be returned as series
        block = block.stack()
        block = block.drop_levels([block.index_columns[0]])
    return block


def _mean_delta_to_power(
    block: blocks.Block,
    n_power,
    column_ids: typing.Sequence[str],
    grouping_column_ids: typing.Sequence[str],
) -> typing.Tuple[blocks.Block, typing.Sequence[str]]:
    """Calculate (x-mean(x))^n. Useful for calculating moment statistics such as skew and kurtosis."""
    window = core.WindowSpec(grouping_keys=grouping_column_ids)
    block, mean_ids = block.multi_apply_window_op(column_ids, agg_ops.mean_op, window)
    delta_ids = []
    cube_op = ops.partial_right(ops.pow_op, n_power)
    for val_id, mean_val_id in zip(column_ids, mean_ids):
        block, delta_id = block.apply_binary_op(val_id, mean_val_id, ops.sub_op)
        block, delta_power_id = block.apply_unary_op(delta_id, cube_op)
        block = block.drop_columns(delta_id)
        delta_ids.append(delta_power_id)
    return block, delta_ids


def _skew_from_moments_and_count(
    block: blocks.Block, count_id: str, moment3_id: str, var_id: str
) -> typing.Tuple[blocks.Block, str]:
    # Calculate skew using count, third moment and population variance
    # See G1 estimator:
    # https://en.wikipedia.org/wiki/Skewness#Sample_skewness
    block, denominator_id = block.apply_unary_op(
        var_id, ops.partial_right(ops.pow_op, 3 / 2)
    )
    block, base_id = block.apply_binary_op(moment3_id, denominator_id, ops.div_op)
    block, countminus1_id = block.apply_unary_op(
        count_id, ops.partial_right(ops.sub_op, 1)
    )
    block, countminus2_id = block.apply_unary_op(
        count_id, ops.partial_right(ops.sub_op, 2)
    )
    block, adjustment_id = block.apply_binary_op(count_id, countminus1_id, ops.mul_op)
    block, adjustment_id = block.apply_unary_op(
        adjustment_id, ops.partial_right(ops.pow_op, 1 / 2)
    )
    block, adjustment_id = block.apply_binary_op(
        adjustment_id, countminus2_id, ops.div_op
    )
    block, skew_id = block.apply_binary_op(base_id, adjustment_id, ops.mul_op)

    # Need to produce NA if have less than 3 data points
    block, na_cond_id = block.apply_unary_op(count_id, ops.partial_right(ops.ge_op, 3))
    block, skew_id = block.apply_binary_op(
        skew_id, na_cond_id, ops.partial_arg3(ops.where_op, None)
    )
    return block, skew_id
