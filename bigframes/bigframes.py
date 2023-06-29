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

"""BigQuery DataFrame top level APIs."""

import typing

from bigframes.dataframe import DataFrame
from bigframes.series import Series


def concat(
    objs: typing.Iterable[typing.Union[DataFrame, Series]],
    *,
    join: typing.Literal["inner", "outer"] = "outer",
    ignore_index: bool = False
) -> typing.Union[DataFrame, Series]:
    """Concatenate DataFrame or Series objects along rows.

    Note: currently only supports DataFrames with matching types for each column name.
    """
    contains_dataframes = any(isinstance(x, DataFrame) for x in objs)
    if not contains_dataframes:
        # Special case, all series, so align everything into single column even if labels don't match
        series = typing.cast(typing.Iterable[Series], objs)
        names = {s.name for s in series}
        # For series case, labels are stripped if they don't all match
        if len(names) > 1:
            blocks = [s._block.with_column_labels([None]) for s in series]
        else:
            blocks = [s._block for s in series]
        block = blocks[0].concat(blocks[1:], how=join, ignore_index=ignore_index)
        return Series(block)
    blocks = [obj._block for obj in objs]
    block = blocks[0].concat(blocks[1:], how=join, ignore_index=ignore_index)
    return DataFrame(block)
