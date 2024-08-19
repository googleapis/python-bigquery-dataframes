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

import numpy as np
import pandas as pd
import pytest

import bigframes.bigquery as bbq
import bigframes.dataframe
import bigframes.pandas as bpd
import bigframes.series as series


def test_struct_from_dataframe():
    srs = series.Series(
        [
            {"version": 1, "project": "pandas"},
            {"version": 2, "project": "pandas"},
            {"version": 1, "project": "numpy"},
        ],
    )
    pd.testing.assert_series_equal(
        srs.to_pandas(),
        bbq.struct(srs.struct.explode()).to_pandas(),
        check_index_type=False,
        check_dtype=False,
    )
