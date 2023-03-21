from datetime import datetime

from google.api_core.exceptions import NotFound
from google.cloud import functions_v2
import ibis.expr.datatypes as dt
import pandas
import pytest

from bigframes import get_remote_function_locations, remote_function


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
def cleanup_cloud_functions(bigquery_client, functions_client):
    """Clean up stale cloud functions."""
    _, location = get_remote_function_locations(bigquery_client.location)
    parent = f"projects/{bigquery_client.project}/locations/{location}"
    request = functions_v2.ListFunctionsRequest(parent=parent)
    page_result = functions_client.list_functions(request=request)
    bigframes_cf_prefix = parent + "/functions/bigframes-"
    for response in page_result:
        if not response.name.startswith(bigframes_cf_prefix):
            continue
        age = datetime.now() - datetime.fromtimestamp(response.update_time.timestamp())
        if age.days <= 0:
            continue
        request = functions_v2.DeleteFunctionRequest(name=response.name)
        try:
            functions_client.delete_function(request=request)
        except NotFound:
            # This can happen when multiple pytest sessions are running in
            # parallel. Two or more sessions may discover the same cloud
            # function, but only one of them would be able to delete it
            # successfully, while the other instance will run into this
            # exception. Ignore this exception.
            pass


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
    table = table.filter(table[col_name].notnull()).head(10)
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
    table = table.filter(table[col_name].notnull()).head(10)
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


def test_remote_function_with_bigframes_series(
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
    # d_int64_col_filtered.dtype is Int64Dtype()
    # d_int64_col_filtered.apply(lambda x: x * x).dtype is int64
    # skip type check for now
    pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)
