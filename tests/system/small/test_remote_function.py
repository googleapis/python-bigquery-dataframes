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

import ibis.expr.datatypes as dt
import pandas
import pytest

from bigframes import remote_function


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection to invoke cloud function for bigframes-dev
    $ bq show --connection --location=us --project_id=bigframes-dev bigframes-rf-conn
    """
    return "bigframes-rf-conn"


def test_remote_function_with_bigframes_series(
    scalars_dfs, bigquery_client, dataset_id_permanent, bq_cf_connection
):
    @remote_function(
        [dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id_permanent,
        bq_cf_connection,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result = bf_int64_col_filtered.apply(square).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result = pd_int64_col_filtered.apply(lambda x: x * x)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # d_int64_col_filtered.dtype is Int64Dtype()
    # d_int64_col_filtered.apply(lambda x: x * x).dtype is int64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)
