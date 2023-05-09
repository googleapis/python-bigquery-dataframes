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

"""Build composite transformers on heterogenous data. This module is styled
after Scikit-Learn's compose module:
https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose"""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.preprocessing


class ColumnTransformer(bigframes.ml.api_primitives.BaseEstimator):
    def __init__(
        self,
        transformers: List[
            Tuple[
                str, bigframes.ml.preprocessing.PreprocessorType, Union[str, List[str]]
            ]
        ],
    ):
        # flatten to per-column
        self._transformers: List[
            Tuple[str, bigframes.ml.preprocessing.PreprocessorType, str]
        ] = []
        for entry in transformers:
            name, transformer, column_or_columns = entry
            if isinstance(column_or_columns, str):
                self._transformers.append((name, transformer, column_or_columns))
            else:
                for column in column_or_columns:
                    self._transformers.append((name, transformer, column))

    @property
    def transformers_(
        self,
    ) -> List[Tuple[str, bigframes.ml.preprocessing.PreprocessorType, str]]:
        """The collection of transformers as tuples of (name, transformer, column)"""
        return self._transformers

    def fit(self, X: bigframes.DataFrame, y: Optional[bigframes.DataFrame] = None):
        raise NotImplementedError("Not supported outside of Pipeline")

    def transform(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        raise NotImplementedError("Not supported outside of Pipeline")
