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

"""Plaintext display representations."""

from __future__ import annotations

import typing
from typing import Union

if typing.TYPE_CHECKING:
    import pandas as pd

    import bigframes.dataframe
    import bigframes.series


def create_text_representation(
    obj: Union[bigframes.dataframe.DataFrame, bigframes.series.Series],
    pandas_df: pd.DataFrame,
    total_rows: typing.Optional[int],
) -> str:
    """Create a text representation of the DataFrame or Series."""
    # TODO(swast): This module should probably just be removed and combined
    # with the html module.
    return obj._create_text_representation(pandas_df, total_rows)
