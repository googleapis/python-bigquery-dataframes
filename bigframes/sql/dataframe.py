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

import dataclasses
from typing import Optional, TYPE_CHECKING, Union

import pandas

from bigframes.core import array_value, bigframe_node, blocks, expression
from bigframes.enums import DefaultIndexKind
import bigframes.operations.aggregations as agg_ops
from bigframes.sql.column import Column
from bigframes.sql.functions import col, expr
from bigframes.sql.resolver import resolve_sql_exprs

if TYPE_CHECKING:
    import bigframes.dataframe
    import bigframes.session


@dataclasses.dataclass(frozen=True)
class DataFrame:
    # Maybe not fully resolved due to sql fragments, we resolve these lazily to improve interactivity
    _value: array_value.ArrayValue

    # temproary, until figure out how to add spark methods to main session object nicely
    @classmethod
    def from_table(
        cls, table: str, session: Optional[bigframes.session.Session] = None
    ):
        # Slightly insane way to do this, but reuses existing snapshotting logic
        import bigframes

        session = session or bigframes.get_global_session()
        return cls(
            _value=session._loader.read_gbq_table(
                table, index_col=DefaultIndexKind.NULL, force_total_order=False
            )._block._expr
        )

    @property
    def columns(self) -> list[str]:
        # Problem: this gets fields which looks for dtype as well
        # TODO: Make node ids retrievable independently of dtypes.
        return list(self._value.column_ids)

    @property
    def _resolved_plan(self) -> bigframe_node.BigFrameNode:
        return resolve_sql_exprs(self._value.node, session=self._value.session)

    def toPandas(self) -> pandas.DataFrame:
        result = self._value.session._executor.execute(
            array_value.ArrayValue(self._resolved_plan)
        )
        return result.to_pandas()

    def to_bigframes(self) -> bigframes.dataframe.DataFrame:
        import bigframes.dataframe

        block = blocks.Block(
            array_value.ArrayValue(self._resolved_plan),
            column_labels=self.columns,
            index_columns=[],
        )
        return bigframes.dataframe.DataFrame(block)

    def select(self, *cols: Union[str, Column]) -> DataFrame:
        col_objs: list[Column] = []
        for column in cols:
            if isinstance(column, str):
                col_objs.append(col(column))
            else:
                col_objs.append(column)
        value, names = self._value.compute_values(
            tuple(column._to_bf_expr() for column in col_objs)
        )
        value = value.select_columns(names)
        value = value.rename_columns(
            {auto_name: column._alias for auto_name, column in zip(names, col_objs)}
        )
        return DataFrame(value)

    def filter(self, condition: Union[str, Column]) -> DataFrame:
        if isinstance(condition, str):
            predicate = expr(condition)
        else:
            predicate = condition
        return DataFrame(self._value.filter(predicate._to_bf_expr()))

    where = filter

    def groupby(self, *cols):
        keys = []
        for part in cols:
            if isinstance(part, str):
                keys.append(col(part))
            if isinstance(part, int):
                keys.append(col(self.columns[part]))
            if isinstance(part, Column):
                keys.append(part)
            else:
                raise NotImplementedError(f"Unsupported grouping key: {col}")
        return GroupedData(self, tuple(keys))

    def __getattr__(self, key: str):
        if key in self.columns:
            return col(key)

        raise AttributeError(key)


@dataclasses.dataclass(frozen=True)
class GroupedData:
    _df: DataFrame
    _keys: tuple[Column, ...]

    def sum(self, *cols: Union[str, Column]) -> DataFrame:
        sum_cols = [col(x) if isinstance(x, str) else x for x in cols]
        frame = self._df.select(*(*self._keys, *sum_cols))

        result_val = frame._value.aggregate(
            aggregations=tuple(
                (
                    expression.UnaryAggregation(
                        agg_ops.SumOp(), arg=expression.deref(sum_col._alias)
                    ),
                    sum_col._alias,
                )
                for sum_col in sum_cols
            ),
            by_column_ids=tuple(key._alias for key in self._keys),
            dropna=False,
        )
        return DataFrame(result_val)
