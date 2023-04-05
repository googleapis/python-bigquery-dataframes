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

from datetime import datetime
import math

from google.api_core.exceptions import NotFound, ResourceExhausted
from google.cloud import functions_v2
import ibis.expr.datatypes as dt
import pandas
import pytest

from bigframes import get_remote_function_locations, remote_function

# Use this to control the number of cloud functions being deleted in a single
# test session. This should help soften the spike of the number of mutations per
# minute tracked against a quota limit (default 60) by the Cloud Functions API
# We are running pytest with "-n 20". Let's say each session lasts about a
# minute, so we are setting a limit of 60/20 = 3 deletions per session.
_MAX_NUM_FUNCTIONS_TO_DELETE_PER_SESSION = 3

# Only for testing with global var
_team_pi = "Team Pi"
_team_euler = "Team Euler"


def get_remote_function_end_points(bigquery_client, dataset_id):
    """Get endpoints used by the remote functions in a datset"""
    endpoints = set()
    routines = bigquery_client.list_routines(dataset=dataset_id)
    for routine in routines:
        rf_options = routine._properties.get("remoteFunctionOptions")
        if not rf_options:
            continue
        rf_endpoint = rf_options.get("endpoint")
        if rf_endpoint:
            endpoints.add(rf_endpoint)
    return endpoints


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection to invoke cloud function for bigframes-dev
    $ bq show --connection --location=us --project_id=bigframes-dev bigframes-rf-conn
    """
    return "bigframes-rf-conn"


@pytest.fixture(scope="module")
def functions_client() -> functions_v2.FunctionServiceClient:
    """Cloud Functions client"""
    return functions_v2.FunctionServiceClient()


@pytest.fixture(scope="module", autouse=True)
def cleanup_cloud_functions(bigquery_client, functions_client, dataset_id_permanent):
    """Clean up stale cloud functions."""
    _, location = get_remote_function_locations(bigquery_client.location)
    parent = f"projects/{bigquery_client.project}/locations/{location}"
    request = functions_v2.ListFunctionsRequest(parent=parent)
    page_result = functions_client.list_functions(request=request)
    bigframes_cf_prefix = parent + "/functions/bigframes-"
    permanent_endpoints = get_remote_function_end_points(
        bigquery_client, dataset_id_permanent
    )
    delete_count = 0
    for response in page_result:
        # Ignore non bigframes cloud functions
        if not response.name.startswith(bigframes_cf_prefix):
            continue

        # Ignore bigframes cloud functions referred by the remote functions in
        # the permanent dataset
        if response.service_config.uri in permanent_endpoints:
            continue

        # Ignore the functions less than one day old
        age = datetime.now() - datetime.fromtimestamp(response.update_time.timestamp())
        if age.days <= 0:
            continue

        # Go ahead and delete
        request = functions_v2.DeleteFunctionRequest(name=response.name)
        try:
            functions_client.delete_function(request=request)
            delete_count += 1
            if delete_count >= _MAX_NUM_FUNCTIONS_TO_DELETE_PER_SESSION:
                break
        except NotFound:
            # This can happen when multiple pytest sessions are running in
            # parallel. Two or more sessions may discover the same cloud
            # function, but only one of them would be able to delete it
            # successfully, while the other instance will run into this
            # exception. Ignore this exception.
            pass
        except ResourceExhausted:
            # This can happen if we are hitting GCP limits, e.g.
            # google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
            # for quota metric 'Per project mutation requests' and limit
            # 'Per project mutation requests per minute per region' of service
            # 'cloudfunctions.googleapis.com' for consumer
            # 'project_number:1084210331973'.
            # [reason: "RATE_LIMIT_EXCEEDED" domain: "googleapis.com" ...
            # Let's stop further clean up and leave it to later.
            break


def test_remote_function_multiply_with_ibis(
    scalars_table_id, ibis_client, bigquery_client, dataset_id, bq_cf_connection
):
    @remote_function(
        [dt.int64(), dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )
    def multiply(x, y):
        return x * y

    project_id, dataset_name, table_name = scalars_table_id.split(".")
    if not ibis_client.dataset:
        ibis_client.dataset = dataset_name

    col_name = "int64_col"
    table = ibis_client.tables[table_name]
    table = table.filter(table[col_name].notnull()).order_by("rowindex").head(10)
    pandas_df_orig = table.execute()

    col = table[col_name]
    col_2x = multiply(col, 2).name("int64_col_2x")
    col_square = multiply(col, col).name("int64_col_square")
    table = table.mutate([col_2x, col_square])
    pandas_df_new = table.execute()

    pandas.testing.assert_series_equal(
        pandas_df_orig[col_name] * 2, pandas_df_new["int64_col_2x"], check_names=False
    )

    pandas.testing.assert_series_equal(
        pandas_df_orig[col_name] * pandas_df_orig[col_name],
        pandas_df_new["int64_col_square"],
        check_names=False,
    )


def test_remote_function_stringify_with_ibis(
    scalars_table_id, ibis_client, bigquery_client, dataset_id, bq_cf_connection
):
    @remote_function(
        [dt.int64()],
        dt.str(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )
    def stringify(x):
        return f"I got {x}"

    project_id, dataset_name, table_name = scalars_table_id.split(".")
    if not ibis_client.dataset:
        ibis_client.dataset = dataset_name

    col_name = "int64_col"
    table = ibis_client.tables[table_name]
    table = table.filter(table[col_name].notnull()).order_by("rowindex").head(10)
    pandas_df_orig = table.execute()

    col = table[col_name]
    col_2x = stringify(col).name("int64_str_col")
    table = table.mutate([col_2x])
    pandas_df_new = table.execute()

    pandas.testing.assert_series_equal(
        pandas_df_orig[col_name].apply(lambda x: f"I got {x}"),
        pandas_df_new["int64_str_col"],
        check_names=False,
    )


def test_remote_function_decorator_with_bigframes_series(
    scalars_dfs, bigquery_client, dataset_id, bq_cf_connection
):
    @remote_function(
        [dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
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
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)


def test_remote_function_explicit_with_bigframes_series(
    scalars_dfs, bigquery_client, dataset_id, bq_cf_connection
):
    def add_one(x):
        return x + 1

    remote_add_one = remote_function(
        [dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )(add_one)

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result = bf_int64_col_filtered.apply(remote_add_one).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result = pd_int64_col_filtered.apply(lambda x: add_one(x))

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    # TODO(shobs): Figure why pandas .apply() changes the dtype, e.g.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x).dtype is int64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)


def test_remote_udf_referring_outside_var(
    scalars_dfs, bigquery_client, dataset_id, bq_cf_connection
):
    POSITIVE_SIGN = 1
    NEGATIVE_SIGN = -1
    NO_SIGN = 0

    def sign(num):
        if num > 0:
            return POSITIVE_SIGN
        elif num < 0:
            return NEGATIVE_SIGN
        return NO_SIGN

    remote_sign = remote_function(
        [dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )(sign)

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result = bf_int64_col_filtered.apply(remote_sign).compute()

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result = pd_int64_col_filtered.apply(lambda x: sign(x))

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    # TODO(shobs): Figure why pandas .apply() changes the dtype, e.g.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x).dtype is int64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)


def test_remote_udf_referring_outside_import(
    scalars_dfs, bigquery_client, dataset_id, bq_cf_connection
):
    import math as mymath

    def circumference(radius):
        return 2 * mymath.pi * radius

    remote_circumference = remote_function(
        [dt.float64()],
        dt.float64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )(circumference)

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_float64_col = scalars_df["float64_col"]
    bf_float64_col_filter = bf_float64_col.notnull()
    bf_float64_col_filtered = bf_float64_col[bf_float64_col_filter]
    bf_result = bf_float64_col_filtered.apply(remote_circumference).compute()

    pd_float64_col = scalars_pandas_df["float64_col"]
    pd_float64_col_filter = pd_float64_col.notnull()
    pd_float64_col_filtered = pd_float64_col[pd_float64_col_filter]
    pd_result = pd_float64_col_filtered.apply(lambda x: circumference(x))

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    # TODO(shobs): Figure why pandas .apply() changes the dtype, e.g.
    # pd_float64_col_filtered.dtype is Float64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x).dtype is float64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)


def test_remote_udf_referring_global_var_and_import(
    scalars_dfs, bigquery_client, dataset_id, bq_cf_connection
):
    def find_team(num):
        boundary = (math.pi + math.e) / 2
        if num >= boundary:
            return _team_euler
        return _team_pi

    remote_func = remote_function(
        [dt.float64()],
        dt.string(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
        reuse=False,
    )(find_team)

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_float64_col = scalars_df["float64_col"]
    bf_float64_col_filter = bf_float64_col.notnull()
    bf_float64_col_filtered = bf_float64_col[bf_float64_col_filter]
    bf_result = bf_float64_col_filtered.apply(remote_func).compute()

    pd_float64_col = scalars_pandas_df["float64_col"]
    pd_float64_col_filter = pd_float64_col.notnull()
    pd_float64_col_filtered = pd_float64_col[pd_float64_col_filter]
    pd_result = pd_float64_col_filtered.apply(lambda x: find_team(x))

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    # TODO(shobs): Figure if the dtype mismatch is by design:
    # bf_result.dtype: string[pyarrow]
    # pd_result.dtype: dtype('O')
    # Skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)
