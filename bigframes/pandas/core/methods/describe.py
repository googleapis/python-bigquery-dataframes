# Copyright 2025 Google LLC
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

from bigframes import dataframe, dtypes, series
from bigframes.core import agg_expressions, blocks
from bigframes.core.reshape import api as rs
from bigframes.operations import aggregations

_DEFAULT_DTYPES = (
    dtypes.NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE + dtypes.TEMPORAL_NUMERIC_BIGFRAMES_TYPES
)


def describe(
    input: dataframe.DataFrame | series.Series,
    include: None | typing.Literal["all"],
) -> dataframe.DataFrame | series.Series:
    if isinstance(input, series.Series):
        # Convert the series to a dataframe, describe it, and cast the result back to a series.
        return series.Series(describe(input.to_frame(), include)._block)
    elif not isinstance(input, dataframe.DataFrame):
        raise TypeError(f"Unsupported type: {type(input)}")

    if include is None:
        numeric_df = _select_dtypes(
            input,
            _DEFAULT_DTYPES,
        )
        if len(numeric_df.columns) == 0:
            # Describe eligible non-numeric columns
            return _describe_non_numeric(input)

        # Otherwise, only describe numeric columns
        return _describe_numeric(input)

    elif include == "all":
        numeric_result = _describe_numeric(input)
        non_numeric_result = _describe_non_numeric(input)

        if len(numeric_result.columns) == 0:
            return non_numeric_result
        elif len(non_numeric_result.columns) == 0:
            return numeric_result
        else:
            # Use reindex after join to preserve the original column order.
            return rs.concat(
                [non_numeric_result, numeric_result], axis=1
            )._reindex_columns(input.columns)

    else:
        raise ValueError(f"Unsupported include type: {include}")


def _describe(
    block: blocks.Block,
    columns: typing.Sequence[str],
    include: None | typing.Literal["all"] = None,
    *,
    as_index: bool = True,
    by_col_ids: typing.Sequence[str] = [],
    dropna: bool = False,
) -> blocks.Block:
    stats: list[agg_expressions.Aggregation] = []
    column_labels: list[typing.Hashable] = []

    for col_id in columns:
        label = block.col_id_to_label[col_id]
        dtype = block.expr.get_column_type(col_id)
        if include != "all" and dtype not in _DEFAULT_DTYPES:
            continue
        agg_ops = _get_aggs_for_dtype(dtype)
        stats.extend(op.as_expr(col_id) for op in agg_ops)
        label_tuple = (label,) if block.column_labels.nlevels == 1 else label
        column_labels.extend((*label_tuple, op.name) for op in agg_ops)  # type: ignore

    agg_block, _ = block.aggregate(
        by_column_ids=by_col_ids,
        aggregations=stats,
        dropna=dropna,
        column_labels=pd.Index(column_labels, name=(*block.index.names, None)),
    )
    return agg_block if as_index else agg_block.reset_index(drop=False)


def _get_aggs_for_dtype(dtype) -> list[aggregations.UnaryAggregateOp]:
    if dtype in dtypes.NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE:
        return [
            aggregations.count_op,
            aggregations.mean_op,
            aggregations.std_op,
            aggregations.min_op,
            aggregations.ApproxQuartilesOp(1),
            aggregations.ApproxQuartilesOp(2),
            aggregations.ApproxQuartilesOp(3),
            aggregations.max_op,
        ]
    elif dtype in dtypes.TEMPORAL_NUMERIC_BIGFRAMES_TYPES:
        return [aggregations.count_op]
    elif dtype in [
        dtypes.STRING_DTYPE,
        dtypes.BOOL_DTYPE,
        dtypes.BYTES_DTYPE,
        dtypes.TIME_DTYPE,
    ]:
        return [aggregations.count_op, aggregations.nunique_op]
    else:
        return []


def _describe_numeric(df: dataframe.DataFrame) -> dataframe.DataFrame:
    number_df_result = typing.cast(
        dataframe.DataFrame,
        _select_dtypes(df, dtypes.NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE).agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ),
    )
    temporal_df_result = typing.cast(
        dataframe.DataFrame,
        _select_dtypes(df, dtypes.TEMPORAL_NUMERIC_BIGFRAMES_TYPES).agg(["count"]),
    )

    if len(number_df_result.columns) == 0:
        return temporal_df_result
    elif len(temporal_df_result.columns) == 0:
        return number_df_result
    else:
        import bigframes.core.reshape.api as rs

        original_columns = _select_dtypes(
            df,
            _DEFAULT_DTYPES,
        ).columns

        # Use reindex after join to preserve the original column order.
        return rs.concat(
            [number_df_result, temporal_df_result],
            axis=1,
        )._reindex_columns(original_columns)


def _describe_non_numeric(df: dataframe.DataFrame) -> dataframe.DataFrame:
    return typing.cast(
        dataframe.DataFrame,
        _select_dtypes(
            df,
            [
                dtypes.STRING_DTYPE,
                dtypes.BOOL_DTYPE,
                dtypes.BYTES_DTYPE,
                dtypes.TIME_DTYPE,
            ],
        ).agg(["count", "nunique"]),
    )


def _select_dtypes(
    df: dataframe.DataFrame, dtypes: typing.Sequence[dtypes.Dtype]
) -> dataframe.DataFrame:
    """Selects columns without considering inheritance relationships."""
    columns = [
        col_id
        for col_id, dtype in zip(df._block.value_columns, df._block.dtypes)
        if dtype in dtypes
    ]
    return dataframe.DataFrame(df._block.select_columns(columns))
