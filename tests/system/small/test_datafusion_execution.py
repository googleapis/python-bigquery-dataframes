# Copyright 2026 Google LLC
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

import pytest

import bigframes
import bigframes.bigquery
from bigframes.testing.utils import assert_frame_equal

datafusion = pytest.importorskip("datafusion")


@pytest.fixture(scope="module")
def session_w_datafusion():
    context = bigframes.BigQueryOptions(location="US", enable_datafusion_execution=True)
    session = bigframes.Session(context=context)
    yield session
    session.close()


def test_datafusion_execution_basic(session_w_datafusion, scalars_pandas_df_index):
    execution_count_before = session_w_datafusion._metrics.execution_count
    bf_df = session_w_datafusion.read_pandas(scalars_pandas_df_index)

    # Test projection and arithmetic
    # int64_too is a column that likely exists, let's verify or use a generic one
    # scalars_pandas_df_index usually has 'int64_too' based on polars test
    bf_df["int64_plus_one"] = bf_df["int64_too"] + 1
    bf_result = bf_df[["int64_plus_one"]].to_pandas()

    # Pandas result
    pd_df = scalars_pandas_df_index.copy()
    pd_df["int64_plus_one"] = pd_df["int64_too"] + 1
    pd_result = pd_df[["int64_plus_one"]]

    # Verify execution stayed local (metrics count shouldn't increase for BQ jobs)
    assert session_w_datafusion._metrics.execution_count == execution_count_before
    assert_frame_equal(bf_result, pd_result)


def test_datafusion_execution_filter(session_w_datafusion, scalars_pandas_df_index):
    execution_count_before = session_w_datafusion._metrics.execution_count
    bf_df = session_w_datafusion.read_pandas(scalars_pandas_df_index)

    # Test filter
    bf_filtered = bf_df[bf_df["int64_too"] > 0]
    bf_result = bf_filtered[["int64_too"]].to_pandas()

    pd_df = scalars_pandas_df_index.copy()
    pd_filtered = pd_df[pd_df["int64_too"] > 0]
    pd_result = pd_filtered[["int64_too"]]

    assert session_w_datafusion._metrics.execution_count == execution_count_before
    assert_frame_equal(bf_result, pd_result)
