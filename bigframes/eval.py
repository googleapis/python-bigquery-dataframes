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

import dataclasses
from typing import Optional

import pandas
import pandas.core.computation.parsing as pandas_eval_parsing

import bigframes.dataframe as dataframe
import bigframes.dtypes
import bigframes.series as series


def eval(df: dataframe.DataFrame, expr: str, target: Optional[dataframe.DataFrame]):
    index_resolver = {
        pandas_eval_parsing.clean_column_name(str(name)): EvalSeries(
            df.index.get_level_values(level).to_series()
        )
        for level, name in enumerate(df.index.names)
    }
    column_resolver = {
        pandas_eval_parsing.clean_column_name(str(name)): EvalSeries(series)
        for name, series in df.items()
    }
    return pandas.eval(
        expr=expr, level=3, target=target, resolvers=(index_resolver, column_resolver)  # type: ignore
    )


@dataclasses.dataclass
class FakeNumpyArray:
    dtype: bigframes.dtypes.Dtype


class EvalSeries(series.Series):
    """Slight modified series that works better with pandas.eval"""

    def __init__(self, underlying: series.Series):
        super().__init__(data=underlying._block)

    @property
    def values(self):
        """Returns fake numpy array with only dtype property so that eval can determine schema without actually downloading the data."""
        return FakeNumpyArray(self.dtype)
