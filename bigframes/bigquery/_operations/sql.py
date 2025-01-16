# Copyright 2024 Google LLC
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

"""SQL escape hatch features."""

from __future__ import annotations

from typing import Sequence

import bigframes.dataframe
import bigframes.series


def sql_scalar(
    sql_template: str,
    columns: Sequence[bigframes.series.Series],
) -> bigframes.series.Series:
    """Create a Series from a SQL template.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.bigquery as bbq
        >>> bpd.options.display.progress_bar = None

        >>> s = bpd.Series([1.5, 2.5, 3.5])
        >>> bbq.sql_scalar("ROUND({0}, 0, 'ROUND_HALF_EVEN')", [s])
        0    2.0
        1    2.0
        2    4.0
        dtype: Float64

    Args:
        sql_template (str):
            A SQL format string with Python-style {0} placeholders for each of
            the Series objects in ``columns``.
        columns (Sequence[bigframes.pandas.Series]):
            Series objects representing the column inputs to the
            ``sql_template``. Must contain at least one Series.

    Returns:
        bigframes.pandas.Series:
            A Series with the SQL applied.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if len(columns) == 0:
        raise ValueError("Must provide at least one column in columns")

    base_series = columns[0]
    return base_series._apply_nary_op(
        # TODO: SqlScalarOp
    )
