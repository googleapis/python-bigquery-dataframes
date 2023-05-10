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
import pandas as pd
import pytest

import bigframes
from bigframes.remote_function import remote_function
from tests.system.utils import assert_pandas_df_equal_ignore_ordering


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection to invoke cloud function for bigframes-dev
    $ bq show --connection --location=us --project_id=bigframes-dev bigframes-rf-conn
    """
    return "bigframes-rf-conn"


@pytest.fixture(scope="module")
def session_with_bq_connection(bq_cf_connection) -> bigframes.Session:
    return bigframes.Session(bigframes.Context(bigquery_connection=bq_cf_connection))


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_remote_function_direct_no_session_param(
    bigquery_client, scalars_dfs, dataset_id_permanent, bq_cf_connection
):
    @remote_function(
        [dt.int64()],
        dt.int64(),
        bigquery_client=bigquery_client,
        dataset=dataset_id_permanent,
        bigquery_connection=bq_cf_connection,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = bf_int64_col.to_frame().assign(result=bf_result_col).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_remote_function_direct_session_param(session_with_bq_connection, scalars_dfs):
    @remote_function(
        [dt.int64()],
        dt.int64(),
        session=session_with_bq_connection,
    )
    def square(x):
        return x * x

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = bf_int64_col.to_frame().assign(result=bf_result_col).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_remote_function_via_session_default(session_with_bq_connection, scalars_dfs):
    # Session has bigquery connection initialized via context. Without an
    # explicit dataset the default dataset from the session would be used.
    # Without an explicit bigquery connection, the one present in Session set
    # through the explicit Context would be used. Without an explicit `reuse`
    # the default behavior of reuse=True will take effect. Please note that the
    # udf is same as the one used in other tests in this file so the underlying
    # cloud function would be common and quickly reused.
    @session_with_bq_connection.remote_function([dt.int64()], dt.int64())
    def square(x):
        return x * x

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = bf_int64_col.to_frame().assign(result=bf_result_col).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_remote_function_via_session_with_overrides(
    session, scalars_dfs, dataset_id_permanent, bq_cf_connection
):
    @session.remote_function(
        [dt.int64()],
        dt.int64(),
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
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = bf_int64_col.to_frame().assign(result=bf_result_col).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_remote_function_via_session_context_connection_setter(
    scalars_dfs, dataset_id, bq_cf_connection
):
    # Creating a session scoped only to this test as we would be setting a
    # property in it
    context = bigframes.Context()
    context.bigquery_connection = bq_cf_connection
    session = bigframes.connect(context)

    # Without an explicit bigquery connection, the one present in Session,
    # set via context setter would be used. Without an explicit `reuse` the
    # default behavior of reuse=True will take effect. Please note that the
    # udf is same as the one used in other tests in this file so the underlying
    # cloud function would be common with reuse=True. Since we are using a
    # unique dataset_id, even though the cloud function would be reused, the bq
    # remote function would still be created, making use of the bq connection
    # set in the Context above.
    @session.remote_function([dt.int64()], dt.int64(), dataset=dataset_id)
    def square(x):
        return x * x

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = bf_int64_col.to_frame().assign(result=bf_result_col).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)
