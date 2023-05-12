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

"""Implements Scikit-Learn's sklearn.preprocessing API"""

from typing import Optional, Union

import pandas

import bigframes
import bigframes.ml
import bigframes.ml.sql


# TODO(bmil): implement a preprocessor class in core.py that standardizes
# compilation to SQL
class StandardScaler(bigframes.ml.api_primitives.BaseEstimator):
    """Implementation of sklearn.preprocessing.StandardScaler.

    When used in a Pipeline, this class will compile to a ML.STANDARD_SCALER and be
    wrapped in a BQML TRANSFORM clause.

    When used outside of a Pipeline, the current implementation will not use BQML.
    This might be changed later to produce a transform-only model instead."""

    def __init__(self):
        # TODO(bmil): remove pandas dependency once BigFrames supports these
        self._avg: Optional[pandas.Series] = None
        self._stdev: Optional[pandas.Series] = None

    def fit(self, X: bigframes.DataFrame):
        # TODO(bmil): ensure columns are numeric
        # TODO(bmil): record schema, and check it matches in .transform
        # TODO(bmil): remove pandas dependency once BigFrames supports these
        pd_X = X.to_pandas()
        self._avg = pd_X.mean()
        self._stdev = pd_X.std()

    def transform(self, X: bigframes.DataFrame):
        if self._avg is None or self._stdev is None:
            raise RuntimeError("A transform must be fitted before .transform()")

        # TODO(bmil): remove pandas dependency once BigFrames supports these
        session = X._block.expr._session
        pd_X = X.to_pandas()
        df = (pd_X - self._avg) / self._stdev
        return session.read_pandas(df)


class OneHotEncoder(bigframes.ml.api_primitives.BaseEstimator):
    """Encodes categorical data in a one-hot-encoding format. Note that this
    method deviates from Scikit-Learn; instead of producing sparse binary columns,
    a the encoding is a single column of STRUCT<index INT64, value DOUBLE>

    When used in a Pipeline, this class will compile to a ML.ONE_HOT_ENCODER and
    be wrapped in a BQML TRANSFORM clause."""

    # All estimators must implement __init__ to document their parameters, even
    # if they don't have any
    def __init__(self):
        pass

    def fit(self, X: bigframes.DataFrame):
        # TODO(bmil): provide a non-pipeline implementation of OneHotEncoder
        raise NotImplementedError("Not yet supported outside of Pipeline")

    def transform(self, X: bigframes.DataFrame):
        # TODO(bmil): provide a non-pipeline implementation of OneHotEncoder
        raise NotImplementedError("Not yet supported outside of Pipeline")


PreprocessorType = Union[StandardScaler, OneHotEncoder]
