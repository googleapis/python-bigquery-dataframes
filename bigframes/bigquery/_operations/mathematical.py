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

from typing import Union

from bigframes import dataframe, dtypes
from bigframes import operations as ops
from bigframes import series


def rand(input_data: Union[series.Series, dataframe.DataFrame]) -> series.Series:
    """
    Generates a pseudo-random value of type FLOAT64 in the range of [0, 1),
    inclusive of 0 and exclusive of 1.

    .. warning::
        This method introduces non-determinism to the expression. Reading the
        same column twice may result in different results.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.bigquery as bbq
        >>> df = bpd.DataFrame({"a": [1, 2, 3]})
        >>> df['random'] = bbq.rand(df)
        >>> # Resulting column 'random' will contain random floats between 0 and 1.

    Args:
        input_data (bigframes.pandas.Series or bigframes.pandas.DataFrame):
            A Series or DataFrame to determine the number of rows and the index
            of the result. The actual values in this input are ignored.

    Returns:
        bigframes.pandas.Series: A new Series of random float values.
    """
    if isinstance(input_data, dataframe.DataFrame):
        if len(input_data.columns) == 0:
            raise ValueError("Input DataFrame must have at least one column.")
        # Use the first column as anchor
        anchor = input_data.iloc[:, 0]
    elif isinstance(input_data, series.Series):
        anchor = input_data
    else:
        raise TypeError(
            f"Unsupported type {type(input_data)}. "
            "Expected bigframes.pandas.Series or bigframes.pandas.DataFrame."
        )

    op = ops.SqlScalarOp(
        _output_type=dtypes.FLOAT_DTYPE,
        sql_template="RAND()",
    )
    return anchor._apply_nary_op(op, [])
