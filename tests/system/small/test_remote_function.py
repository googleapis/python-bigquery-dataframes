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

import re

import google.api_core.exceptions
from google.cloud import bigquery
import pandas as pd
import pytest

import bigframes
import bigframes.exceptions
from bigframes.functions import remote_function as rf
from tests.system.utils import assert_pandas_df_equal


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection in the test project in US location, used to
    invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "bigframes-rf-conn"


@pytest.fixture(scope="module")
def bq_cf_connection_location() -> str:
    """Pre-created BQ connection in the test project in US location, in format
    PROJECT_ID.LOCATION.CONNECTION_NAME, used to invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "us.bigframes-rf-conn"


@pytest.fixture(scope="module")
def bq_cf_connection_location_mismatched() -> str:
    """Pre-created BQ connection in the test project in EU location, in format
    LOCATION.CONNECTION_NAME, used to invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "eu.bigframes-rf-conn"


@pytest.fixture(scope="module")
def bq_cf_connection_location_project(bigquery_client) -> str:
    """Pre-created BQ connection in the test project in US location, in format
    PROJECT_ID.LOCATION.CONNECTION_NAME, used to invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return f"{bigquery_client.project}.us.bigframes-rf-conn"


@pytest.fixture(scope="module")
def bq_cf_connection_location_project_mismatched() -> str:
    """Pre-created BQ connection in the bigframes-metrics project in US location,
    in format PROJECT_ID.LOCATION.CONNECTION_NAME, used to invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "bigframes-metrics.eu.bigframes-rf-conn"


@pytest.fixture(scope="module")
def session_with_bq_connection(
    bq_cf_connection, dataset_id_permanent
) -> bigframes.Session:
    session = bigframes.Session(
        bigframes.BigQueryOptions(bq_connection=bq_cf_connection)
    )
    return session


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_no_session_param(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    scalars_dfs,
    dataset_id_permanent,
    bq_cf_connection,
):
    @rf.remote_function(
        [int],
        int,
        bigquery_client=bigquery_client,
        bigquery_connection_client=bigqueryconnection_client,
        cloud_functions_client=cloudfunctions_client,
        resource_manager_client=resourcemanager_client,
        dataset=dataset_id_permanent,
        bigquery_connection=bq_cf_connection,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    # Function should have extra metadata attached for remote execution.
    assert hasattr(square, "bigframes_remote_function")
    assert hasattr(square, "bigframes_cloud_function")
    assert hasattr(square, "ibis_node")

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_no_session_param_location_specified(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    scalars_dfs,
    dataset_id_permanent,
    bq_cf_connection_location,
):
    @rf.remote_function(
        [int],
        int,
        bigquery_client=bigquery_client,
        bigquery_connection_client=bigqueryconnection_client,
        cloud_functions_client=cloudfunctions_client,
        resource_manager_client=resourcemanager_client,
        dataset=dataset_id_permanent,
        bigquery_connection=bq_cf_connection_location,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_no_session_param_location_mismatched(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    dataset_id_permanent,
    bq_cf_connection_location_mismatched,
):
    with pytest.raises(
        ValueError,
        match=re.escape("The location does not match BigQuery connection location:"),
    ):

        @rf.remote_function(
            [int],
            int,
            bigquery_client=bigquery_client,
            bigquery_connection_client=bigqueryconnection_client,
            cloud_functions_client=cloudfunctions_client,
            resource_manager_client=resourcemanager_client,
            dataset=dataset_id_permanent,
            bigquery_connection=bq_cf_connection_location_mismatched,
            # See e2e tests for tests that actually deploy the Cloud Function.
            reuse=True,
        )
        def square(x):
            # Not expected to reach this code, as the location of the
            # connection doesn't match the location of the dataset.
            return x * x  # pragma: NO COVER


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_no_session_param_location_project_specified(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    scalars_dfs,
    dataset_id_permanent,
    bq_cf_connection_location_project,
):
    @rf.remote_function(
        [int],
        int,
        bigquery_client=bigquery_client,
        bigquery_connection_client=bigqueryconnection_client,
        cloud_functions_client=cloudfunctions_client,
        resource_manager_client=resourcemanager_client,
        dataset=dataset_id_permanent,
        bigquery_connection=bq_cf_connection_location_project,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_no_session_param_project_mismatched(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    dataset_id_permanent,
    bq_cf_connection_location_project_mismatched,
):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The project_id does not match BigQuery connection gcp_project_id:"
        ),
    ):

        @rf.remote_function(
            [int],
            int,
            bigquery_client=bigquery_client,
            bigquery_connection_client=bigqueryconnection_client,
            cloud_functions_client=cloudfunctions_client,
            resource_manager_client=resourcemanager_client,
            dataset=dataset_id_permanent,
            bigquery_connection=bq_cf_connection_location_project_mismatched,
            # See e2e tests for tests that actually deploy the Cloud Function.
            reuse=True,
        )
        def square(x):
            # Not expected to reach this code, as the project of the
            # connection doesn't match the project of the dataset.
            return x * x  # pragma: NO COVER


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_direct_session_param(session_with_bq_connection, scalars_dfs):
    @rf.remote_function(
        [int],
        int,
        session=session_with_bq_connection,
    )
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_via_session_default(session_with_bq_connection, scalars_dfs):
    # Session has bigquery connection initialized via context. Without an
    # explicit dataset the default dataset from the session would be used.
    # Without an explicit bigquery connection, the one present in Session set
    # through the explicit BigQueryOptions would be used. Without an explicit `reuse`
    # the default behavior of reuse=True will take effect. Please note that the
    # udf is same as the one used in other tests in this file so the underlying
    # cloud function would be common and quickly reused.
    @session_with_bq_connection.remote_function([int], int)
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_remote_function_via_session_with_overrides(
    session, scalars_dfs, dataset_id_permanent, bq_cf_connection
):
    @session.remote_function(
        [int],
        int,
        dataset_id_permanent,
        bq_cf_connection,
        # See e2e tests for tests that actually deploy the Cloud Function.
        reuse=True,
    )
    def square(x):
        return x * x

    # Function should still work normally.
    assert square(2) == 4

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    # TODO(shobs): Figure why pandas .apply() changes the dtype, i.e.
    # pd_int64_col_filtered.dtype is Int64Dtype()
    # pd_int64_col_filtered.apply(lambda x: x * x).dtype is int64.
    # For this test let's force the pandas dtype to be same as bigframes' dtype.
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_dataframe_applymap(session_with_bq_connection, scalars_dfs):
    def add_one(x):
        return x + 1

    remote_add_one = session_with_bq_connection.remote_function([int], int)(add_one)

    scalars_df, scalars_pandas_df = scalars_dfs
    int64_cols = ["int64_col", "int64_too"]

    bf_int64_df = scalars_df[int64_cols]
    bf_int64_df_filtered = bf_int64_df.dropna()
    bf_result = bf_int64_df_filtered.applymap(remote_add_one).to_pandas()

    pd_int64_df = scalars_pandas_df[int64_cols]
    pd_int64_df_filtered = pd_int64_df.dropna()
    pd_result = pd_int64_df_filtered.applymap(add_one)
    # TODO(shobs): Figure why pandas .applymap() changes the dtype, i.e.
    # pd_int64_df_filtered.dtype is Int64Dtype()
    # pd_int64_df_filtered.applymap(lambda x: x).dtype is int64.
    # For this test let's force the pandas dtype to be same as input.
    for col in pd_result:
        pd_result[col] = pd_result[col].astype(pd_int64_df_filtered[col].dtype)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_dataframe_applymap_na_ignore(session_with_bq_connection, scalars_dfs):
    def add_one(x):
        return x + 1

    remote_add_one = session_with_bq_connection.remote_function([int], int)(add_one)

    scalars_df, scalars_pandas_df = scalars_dfs
    int64_cols = ["int64_col", "int64_too"]

    bf_int64_df = scalars_df[int64_cols]
    bf_result = bf_int64_df.applymap(remote_add_one, na_action="ignore").to_pandas()

    pd_int64_df = scalars_pandas_df[int64_cols]
    pd_result = pd_int64_df.applymap(add_one, na_action="ignore")
    # TODO(shobs): Figure why pandas .applymap() changes the dtype, i.e.
    # pd_int64_df_filtered.dtype is Int64Dtype()
    # pd_int64_df_filtered.applymap(lambda x: x).dtype is int64.
    # For this test let's force the pandas dtype to be same as input.
    for col in pd_result:
        pd_result[col] = pd_result[col].astype(pd_int64_df[col].dtype)

    assert_pandas_df_equal(bf_result, pd_result)


@pytest.mark.flaky(retries=2, delay=120)
def test_series_map(session_with_bq_connection, scalars_dfs):
    def add_one(x):
        return x + 1

    remote_add_one = session_with_bq_connection.remote_function([int], int)(add_one)

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_result = scalars_df.int64_too.map(remote_add_one).to_pandas()
    pd_result = scalars_pandas_df.int64_too.map(add_one)
    pd_result = pd_result.astype("Int64")  # pandas type differences

    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
    )


def test_skip_bq_connection_check(dataset_id_permanent):
    connection_name = "connection_does_not_exist"
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(
            bq_connection=connection_name, skip_bq_connection_check=True
        )
    )

    # Make sure that the connection does not exist
    with pytest.raises(google.api_core.exceptions.NotFound):
        session.bqconnectionclient.get_connection(
            name=session.bqconnectionclient.connection_path(
                session._project, session._location, connection_name
            )
        )

    # Make sure that an attempt to create a remote function routine with
    # non-existent connection would result in an exception thrown by the BQ
    # service.
    # This is different from the exception throw by the BQ Connection service
    # if it was not able to create the connection because of lack of permission
    # when skip_bq_connection_check was not set to True:
    # google.api_core.exceptions.PermissionDenied: 403 Permission 'resourcemanager.projects.setIamPolicy' denied on resource
    with pytest.raises(
        google.api_core.exceptions.NotFound,
        match=f"Not found: Connection {connection_name}",
    ):

        @session.remote_function([int], int, dataset=dataset_id_permanent)
        def add_one(x):
            # Not expected to reach this code, as the connection doesn't exist.
            return x + 1  # pragma: NO COVER


@pytest.mark.flaky(retries=2, delay=120)
def test_read_gbq_function_detects_invalid_function(bigquery_client, dataset_id):
    dataset_ref = bigquery.DatasetReference.from_string(dataset_id)
    with pytest.raises(ValueError) as e:
        rf.read_gbq_function(
            str(dataset_ref.routine("not_a_function")),
            bigquery_client=bigquery_client,
        )

    assert "Unknown function" in str(e.value)


@pytest.mark.flaky(retries=2, delay=120)
def test_read_gbq_function_like_original(
    bigquery_client,
    bigqueryconnection_client,
    cloudfunctions_client,
    resourcemanager_client,
    scalars_df_index,
    dataset_id_permanent,
    bq_cf_connection,
):
    @rf.remote_function(
        [int],
        int,
        bigquery_client=bigquery_client,
        bigquery_connection_client=bigqueryconnection_client,
        dataset=dataset_id_permanent,
        cloud_functions_client=cloudfunctions_client,
        resource_manager_client=resourcemanager_client,
        bigquery_connection=bq_cf_connection,
        reuse=True,
    )
    def square1(x):
        return x * x

    # Function should still work normally.
    assert square1(2) == 4

    square2 = rf.read_gbq_function(
        function_name=square1.bigframes_remote_function,
        bigquery_client=bigquery_client,
    )

    # The newly-created function (square1) should have a remote function AND a
    # cloud function associated with it, while the read-back version (square2)
    # should only have a remote function.
    assert square1.bigframes_remote_function
    assert square1.bigframes_cloud_function

    assert square2.bigframes_remote_function
    assert not hasattr(square2, "bigframes_cloud_function")

    # They should point to the same function.
    assert square1.bigframes_remote_function == square2.bigframes_remote_function

    # The result of applying them should be the same.
    int64_col = scalars_df_index["int64_col"]
    int64_col_filter = int64_col.notnull()
    int64_col_filtered = int64_col[int64_col_filter]

    s1_result_col = int64_col_filtered.apply(square1)
    s1_result = int64_col_filtered.to_frame().assign(result=s1_result_col)

    s2_result_col = int64_col_filtered.apply(square2)
    s2_result = int64_col_filtered.to_frame().assign(result=s2_result_col)

    assert_pandas_df_equal(s1_result.to_pandas(), s2_result.to_pandas())


@pytest.mark.flaky(retries=2, delay=120)
def test_read_gbq_function_reads_udfs(bigquery_client, dataset_id):
    dataset_ref = bigquery.DatasetReference.from_string(dataset_id)
    arg = bigquery.RoutineArgument(
        name="x",
        data_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
    )
    sql_routine = bigquery.Routine(
        dataset_ref.routine("square_sql"),
        body="x * x",
        arguments=[arg],
        return_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )
    js_routine = bigquery.Routine(
        dataset_ref.routine("square_js"),
        body="return x * x",
        language="JAVASCRIPT",
        arguments=[arg],
        return_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )

    for routine in (sql_routine, js_routine):
        # Create the routine in BigQuery and read it back using read_gbq_function.
        bigquery_client.create_routine(routine, exists_ok=True)
        square = rf.read_gbq_function(
            str(routine.reference), bigquery_client=bigquery_client
        )

        # It should point to the named routine and yield the expected results.
        assert square.bigframes_remote_function == str(routine.reference)

        src = {"x": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]}

        routine_ref_str = rf.routine_ref_to_string_for_query(routine.reference)
        direct_sql = " UNION ALL ".join(
            [f"SELECT {x} AS x, {routine_ref_str}({x}) AS y" for x in src["x"]]
        )
        direct_df = bigquery_client.query(direct_sql).to_dataframe()

        indirect_df = bigframes.dataframe.DataFrame(src)
        indirect_df = indirect_df.assign(y=indirect_df.x.apply(square))
        # TODO(b/340875260): fix type error
        indirect_df = indirect_df.to_pandas()  # type: ignore

        assert_pandas_df_equal(
            direct_df, indirect_df, ignore_order=True, check_index_type=False
        )


@pytest.mark.flaky(retries=2, delay=120)
def test_read_gbq_function_enforces_explicit_types(bigquery_client, dataset_id):
    dataset_ref = bigquery.DatasetReference.from_string(dataset_id)
    typed_arg = bigquery.RoutineArgument(
        name="x",
        data_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
    )
    untyped_arg = bigquery.RoutineArgument(
        name="x",
        kind="ANY_TYPE",  # With this kind, data_type not required for SQL functions.
    )

    both_types_specified = bigquery.Routine(
        dataset_ref.routine("both_types_specified"),
        body="x * x",
        arguments=[typed_arg],
        return_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )
    only_return_type_specified = bigquery.Routine(
        dataset_ref.routine("only_return_type_specified"),
        body="x * x",
        arguments=[untyped_arg],
        return_type=bigquery.StandardSqlDataType(bigquery.StandardSqlTypeNames.INT64),
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )
    only_arg_type_specified = bigquery.Routine(
        dataset_ref.routine("only_arg_type_specified"),
        body="x * x",
        arguments=[typed_arg],
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )
    neither_type_specified = bigquery.Routine(
        dataset_ref.routine("neither_type_specified"),
        body="x * x",
        arguments=[untyped_arg],
        type_=bigquery.RoutineType.SCALAR_FUNCTION,
    )

    bigquery_client.create_routine(both_types_specified, exists_ok=True)
    bigquery_client.create_routine(only_return_type_specified, exists_ok=True)
    bigquery_client.create_routine(only_arg_type_specified, exists_ok=True)
    bigquery_client.create_routine(neither_type_specified, exists_ok=True)

    rf.read_gbq_function(
        str(both_types_specified.reference), bigquery_client=bigquery_client
    )
    rf.read_gbq_function(
        str(only_return_type_specified.reference), bigquery_client=bigquery_client
    )
    with pytest.raises(ValueError):
        rf.read_gbq_function(
            str(only_arg_type_specified.reference), bigquery_client=bigquery_client
        )
    with pytest.raises(ValueError):
        rf.read_gbq_function(
            str(neither_type_specified.reference), bigquery_client=bigquery_client
        )


@pytest.mark.flaky(retries=2, delay=120)
def test_df_apply_axis_1(session, scalars_dfs):
    columns = ["bool_col", "int64_col", "int64_too", "float64_col", "string_col"]
    scalars_df, scalars_pandas_df = scalars_dfs

    def add_ints(row):
        return row["int64_col"] + row["int64_too"]

    with pytest.warns(
        bigframes.exceptions.PreviewWarning,
        match="input_types=Series is in preview.",
    ):
        add_ints_remote = session.remote_function(bigframes.series.Series, int)(
            add_ints
        )

    with pytest.warns(
        bigframes.exceptions.PreviewWarning, match="axis=1 scenario is in preview."
    ):
        bf_result = scalars_df[columns].apply(add_ints_remote, axis=1).to_pandas()

    pd_result = scalars_pandas_df[columns].apply(add_ints, axis=1)

    # bf_result.dtype is 'Int64' while pd_result.dtype is 'object', ignore this
    # mismatch by using check_dtype=False.
    #
    # bf_result.to_numpy() produces an array of numpy.float64's
    # (in system_prerelease tests), while pd_result.to_numpy() produces an
    # array of ints, ignore this mismatch by using check_exact=False.
    pd.testing.assert_series_equal(
        pd_result, bf_result, check_dtype=False, check_exact=False
    )


@pytest.mark.flaky(retries=2, delay=120)
def test_df_apply_axis_1_ordering(session, scalars_dfs):
    columns = ["bool_col", "int64_col", "int64_too", "float64_col", "string_col"]
    ordering_columns = ["bool_col", "int64_col"]
    scalars_df, scalars_pandas_df = scalars_dfs

    def add_ints(row):
        return row["int64_col"] + row["int64_too"]

    add_ints_remote = session.remote_function(bigframes.series.Series, int)(add_ints)

    bf_result = (
        scalars_df[columns]
        .sort_values(ordering_columns)
        .apply(add_ints_remote, axis=1)
        .to_pandas()
    )
    pd_result = (
        scalars_pandas_df[columns].sort_values(ordering_columns).apply(add_ints, axis=1)
    )

    # bf_result.dtype is 'Int64' while pd_result.dtype is 'object', ignore this
    # mismatch by using check_dtype=False.
    #
    # bf_result.to_numpy() produces an array of numpy.float64's
    # (in system_prerelease tests), while pd_result.to_numpy() produces an
    # array of ints, ignore this mismatch by using check_exact=False.
    pd.testing.assert_series_equal(
        pd_result, bf_result, check_dtype=False, check_exact=False
    )


@pytest.mark.flaky(retries=2, delay=120)
def test_df_apply_axis_1_multiindex(session):
    pd_df = pd.DataFrame(
        {"x": [1, 2, 3], "y": [1.5, 3.75, 5], "z": ["pq", "rs", "tu"]},
        index=pd.MultiIndex.from_tuples([("a", 100), ("a", 200), ("b", 300)]),
    )
    bf_df = session.read_pandas(pd_df)

    def add_numbers(row):
        return row["x"] + row["y"]

    add_numbers_remote = session.remote_function(bigframes.series.Series, float)(
        add_numbers
    )

    bf_result = bf_df.apply(add_numbers_remote, axis=1).to_pandas()
    pd_result = pd_df.apply(add_numbers, axis=1)

    # bf_result.dtype is 'Float64' while pd_result.dtype is 'float64', ignore this
    # mismatch by using check_dtype=False.
    #
    # bf_result.index[0].dtype is 'string[pyarrow]' while
    # pd_result.index[0].dtype is 'object', ignore this mismatch by using
    # check_index_type=False.
    pd.testing.assert_series_equal(
        pd_result, bf_result, check_dtype=False, check_index_type=False
    )


def test_df_apply_axis_1_unsupported_callable(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    columns = ["bool_col", "int64_col", "int64_too", "float64_col", "string_col"]

    def add_ints(row):
        return row["int64_col"] + row["int64_too"]

    # pandas works
    scalars_pandas_df.apply(add_ints, axis=1)

    with pytest.raises(ValueError, match="For axis=1 a remote function must be used."):
        scalars_df[columns].apply(add_ints, axis=1)


@pytest.mark.parametrize(
    ("column"),
    [
        pytest.param("bytes_col"),
        pytest.param("date_col"),
        pytest.param("datetime_col"),
        pytest.param("geography_col"),
        pytest.param("numeric_col"),
        pytest.param("time_col"),
        pytest.param("timestamp_col"),
    ],
)
def test_df_apply_axis_1_unsupported_dtype(scalars_dfs, column):
    scalars_df, scalars_pandas_df = scalars_dfs

    # It doesn't matter if it is a remote function or not, the dtype check
    # is done even before the function type check with axis=1
    def echo(row):
        return row[column]

    # pandas works
    scalars_pandas_df[[column]].apply(echo, axis=1)

    dtype = scalars_df[column].dtype

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"DataFrame has a column of dtype '{dtype}' which is not supported with axis=1. Supported dtypes are ('Int64', 'Float64', 'boolean', 'string')."
        ),
    ):
        scalars_df[[column]].apply(echo, axis=1)
