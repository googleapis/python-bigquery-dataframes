import ibis.expr.datatypes as dt
import pandas

from bigframes import remote_function


def test_remote_function_multiply_with_ibis(
    scalars_table_id, ibis_client, bigquery_client, dataset_id, bq_cf_connection
):
    @remote_function(
        [dt.int64(), dt.int64()],
        dt.int64(),
        bigquery_client,
        dataset_id,
        bq_cf_connection,
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


def test_remote_function_with_bigframes():
    # TODO(shobs): Implement a pandas-like API utilizing remote function and
    # test it here
    pass
