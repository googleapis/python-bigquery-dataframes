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

import pandas as pd


def test_get_dtypes(scalars_df, scalars_pandas_df):
    dtypes = scalars_df.dtypes
    pd.testing.assert_series_equal(
        dtypes,
        scalars_pandas_df.dtypes,
    )


def test_get_columns(scalars_df, scalars_pandas_df):
    pd.testing.assert_index_equal(scalars_df.columns, scalars_pandas_df.columns)


def test_to_sql_query(scalars_df):
    # Note: Exact generated SQL depends on Ibis backend
    # so don't test it here
    sql, _ = scalars_df.to_sql_query(always_include_index=False)
    assert "SELECT " in sql


def test_assign_coerces_literals_to_compatible_types(scalars_df):
    scalars_df = scalars_df.assign(new_int_col=2, new_float_col=3.0)
    assert scalars_df.new_int_col.dtype == pd.Int64Dtype()
    assert scalars_df.new_float_col.dtype == pd.Float64Dtype()
