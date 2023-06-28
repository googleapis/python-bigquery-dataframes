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

import ibis.expr.types as ibis_types
import pandas as pd

import bigframes.core.blocks as blocks
import bigframes.dtypes
import bigframes.operations as ops
import bigframes.series as series
import bigframes.session
import third_party.bigframes_vendored.pandas.pandas._typing as vendored_pandas_typing

# BigQuery has 1 MB query size limit, 5000 items shouldn't take more than 10% of this depending on data type.
# TODO(tbergeron): Convert to bytes-based limit
MAX_INLINE_SERIES_SIZE = 5000


class SeriesMethods:
    def __init__(
        self,
        data=None,
        index: vendored_pandas_typing.Axes | None = None,
        dtype: typing.Optional[
            bigframes.dtypes.DtypeString | bigframes.dtypes.Dtype
        ] = None,
        name: str | None = None,
        copy: typing.Optional[bool] = None,
        *,
        session: typing.Optional[bigframes.session.Session] = None,
    ):
        block = None
        if copy is not None and not copy:
            raise ValueError("Series constructor only supports copy=True")
        if isinstance(data, blocks.Block):
            assert len(data.value_columns) == 1
            assert len(data.column_labels) == 1
            block = data

        elif isinstance(data, SeriesMethods):
            block = data._get_block()

        if block:
            if name:
                if not isinstance(name, str):
                    raise NotImplementedError(
                        "BigQuery DataFrame only supports string series names."
                    )
                block = block.with_column_labels([name])
            if index:
                raise NotImplementedError(
                    "Series 'index' constructor parameter not supported when passing BigQuery-backed objects"
                )
            if dtype:
                block = block.multi_apply_unary_op(
                    block.value_columns, ops.AsTypeOp(dtype)
                )
            self._block = block

        else:
            import bigframes.pandas

            pd_dataframe = pd.Series(
                data=data, index=index, dtype=dtype, name=name  # type:ignore
            ).to_frame()
            if pd_dataframe.size < MAX_INLINE_SERIES_SIZE:
                self._block = blocks.block_from_local(
                    pd_dataframe, session or bigframes.pandas.get_global_session()
                )
            if session:
                self._block = session.read_pandas(pd_dataframe)._get_block()
            else:
                # Uses default global session
                self._block = bigframes.pandas.read_pandas(pd_dataframe)._get_block()

    @property
    def _value(self) -> ibis_types.Value:
        """Private property to get Ibis expression for the value column."""
        return self._block.expr.get_column(self._value_column)

    @property
    def _value_column(self) -> str:
        return self._block.value_columns[0]

    @property
    def _name(self) -> blocks.Label:
        return self._block.column_labels[0]

    def _set_block(self, block: blocks.Block):
        self._block = block

    def _get_block(self) -> blocks.Block:
        return self._block

    def _apply_unary_op(
        self,
        op: ops.UnaryOp,
    ) -> series.Series:
        """Applies a unary operator to the series."""
        block, result_id = self._block.apply_unary_op(
            self._value_column, op, result_label=self._name
        )
        return series.Series(block.select_column(result_id))
