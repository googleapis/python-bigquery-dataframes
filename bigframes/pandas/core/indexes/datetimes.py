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

import inspect

import bigframes.core.global_session as global_session
import bigframes.core.indexes
import bigframes.session


def date_range(
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
) -> bigframes.core.indexes.DatetimeIndex:
    return global_session.with_default_session(
        bigframes.session.Session.date_range,
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        tz=tz,
        normalize=normalize,
        name=name,
        inclusive=inclusive,
        unit=unit,
    )


date_range.__doc__ = inspect.getdoc(bigframes.session.Session.date_range)
