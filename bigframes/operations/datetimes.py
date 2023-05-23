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

import bigframes.operations as ops
import bigframes.operations.base
import bigframes.series as series


class DatetimeMethods(bigframes.operations.base.SeriesMethods):
    """Methods that act on a Datetime Series."""

    @property
    def day(self) -> series.Series:
        """Returns the day of the datetime"""
        return self._apply_unary_op(ops.day_op)

    @property
    def dayofweek(self) -> series.Series:
        """Return the day of the week.
        It is assumed the week starts on Monday,
        which is denoted by 0 and ends on Sunday which is denoted by 6.
        This method is available on both Series with datetime values (using the dt accessor)"""
        return self._apply_unary_op(ops.dayofweek_op)

    @property
    def date(self) -> series.Series:
        """Extracts date from a datetime/timestamp series,

        warning:
            This method returns a Series whereas pandas returns
            a numpy array.
        """
        return self._apply_unary_op(ops.date_op)

    @property
    def hour(self) -> series.Series:
        """Extracts the hours of the datetime."""
        return self._apply_unary_op(ops.hour_op)

    @property
    def minute(self) -> series.Series:
        """Extracts the minutes of the datetime."""
        return self._apply_unary_op(ops.minute_op)

    @property
    def month(self) -> series.Series:
        """Extracts month from a timestamp series"""
        return self._apply_unary_op(ops.month_op)

    @property
    def second(self) -> series.Series:
        """Extracts second from a timestamp series"""
        return self._apply_unary_op(ops.second_op)

    @property
    def time(self) -> series.Series:
        """Extracts time from a datetime/timestamp series,

        warning:
            This method returns a Series whereas pandas returns
            a numpy array.
        """
        return self._apply_unary_op(ops.time_op)

    @property
    def year(self) -> series.Series:
        """Returns the year of the datetime"""
        return self._apply_unary_op(ops.year_op)
