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

"""An index based on a single column with a datetime-like data type."""

from __future__ import annotations

from bigframes_vendored.pandas.core.indexes import (
    datetimes as vendored_pandas_datetime_index,
)
import pandas

from bigframes.core import expression as ex
from bigframes.core.indexes.base import Index
from bigframes.operations import date_ops


class DatetimeIndex(Index, vendored_pandas_datetime_index.DatetimeIndex):
    __doc__ = vendored_pandas_datetime_index.DatetimeIndex.__doc__

    # Must be above 5000 for pandas to delegate to bigframes for binops
    __pandas_priority__ = 12000

    @property
    def year(self) -> Index:
        return self._apply_unary_expr(date_ops.year_op.as_expr(ex.free_var("arg")))

    @property
    def month(self) -> Index:
        return self._apply_unary_expr(date_ops.month_op.as_expr(ex.free_var("arg")))

    @property
    def day(self) -> Index:
        return self._apply_unary_expr(date_ops.day_op.as_expr(ex.free_var("arg")))

    @property
    def dayofweek(self) -> Index:
        return self._apply_unary_expr(date_ops.dayofweek_op.as_expr(ex.free_var("arg")))

    @property
    def day_of_week(self) -> Index:
        return self.dayofweek

    @property
    def weekday(self) -> Index:
        return self.dayofweek


def date_range(
    session,
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name=None,
    inclusive="both",
    *,
    unit: str | None = None,
) -> DatetimeIndex:
    kwargs = {}
    if unit is not None:
        kwargs["unit"] = unit

    pd_index = pandas.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        **kwargs,  # type: ignore
    )

    return session.read_pandas(pd_index)


date_range.__doc__ = vendored_pandas_datetime_index.date_range.__doc__
