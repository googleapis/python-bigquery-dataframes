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

import google.api_core.exceptions
import pandas
import pyarrow
import pytest

import bigframes
import bigframes.pandas as bpd
from tests.system.utils import cleanup_function_assets

bpd.options.experiments.udf = True


def test_managed_function_multiply_with_ibis(
    session,
    scalars_table_id,
    bigquery_client,
    ibis_client,
    dataset_id,
):

    try:

        @session.udf(
            input_types=[int, int],
            output_type=int,
            dataset=dataset_id,
        )
        def multiply(x, y):
            return x * y

        _, dataset_name, table_name = scalars_table_id.split(".")
        if not ibis_client.dataset:
            ibis_client.dataset = dataset_name

        col_name = "int64_col"
        table = ibis_client.tables[table_name]
        table = table.filter(table[col_name].notnull()).order_by("rowindex").head(10)
        sql = table.compile()
        pandas_df_orig = bigquery_client.query(sql).to_dataframe()

        col = table[col_name]
        col_2x = multiply(col, 2).name("int64_col_2x")
        col_square = multiply(col, col).name("int64_col_square")
        table = table.mutate([col_2x, col_square])
        sql = table.compile()
        pandas_df_new = bigquery_client.query(sql).to_dataframe()

        pandas.testing.assert_series_equal(
            pandas_df_orig[col_name] * 2,
            pandas_df_new["int64_col_2x"],
            check_names=False,
        )

        pandas.testing.assert_series_equal(
            pandas_df_orig[col_name] * pandas_df_orig[col_name],
            pandas_df_new["int64_col_square"],
            check_names=False,
        )
    finally:
        # clean up the gcp assets created for the managed function.
        cleanup_function_assets(multiply, bigquery_client)


def test_managed_function_stringify_with_ibis(
    session,
    scalars_table_id,
    bigquery_client,
    ibis_client,
    dataset_id,
):
    try:

        @session.udf(
            input_types=[int],
            output_type=str,
            dataset=dataset_id,
        )
        def stringify(x):
            return f"I got {x}"

        # Function should work locally.
        assert stringify(8912) == "I got 8912"

        _, dataset_name, table_name = scalars_table_id.split(".")
        if not ibis_client.dataset:
            ibis_client.dataset = dataset_name

        col_name = "int64_col"
        table = ibis_client.tables[table_name]
        table = table.filter(table[col_name].notnull()).order_by("rowindex").head(10)
        sql = table.compile()
        pandas_df_orig = bigquery_client.query(sql).to_dataframe()

        col = table[col_name]
        col_2x = stringify.ibis_node(col).name("int64_str_col")
        table = table.mutate([col_2x])
        sql = table.compile()
        pandas_df_new = bigquery_client.query(sql).to_dataframe()

        pandas.testing.assert_series_equal(
            pandas_df_orig[col_name].apply(lambda x: f"I got {x}"),
            pandas_df_new["int64_str_col"],
            check_names=False,
        )
    finally:
        # clean up the gcp assets created for the managed function.
        cleanup_function_assets(stringify, bigquery_client)


@pytest.mark.parametrize(
    "array_dtype",
    [
        bool,
        int,
        float,
        str,
    ],
)
def test_managed_function_array_output(session, scalars_dfs, dataset_id, array_dtype):
    try:

        @session.udf(dataset=dataset_id)
        def featurize(x: int) -> list[array_dtype]:  # type: ignore
            return [array_dtype(i) for i in [x, x + 1, x + 2]]

        scalars_df, scalars_pandas_df = scalars_dfs

        bf_int64_col = scalars_df["int64_too"]
        bf_result = bf_int64_col.apply(featurize).to_pandas()

        pd_int64_col = scalars_pandas_df["int64_too"]
        pd_result = pd_int64_col.apply(featurize)

        # Ignore any dtype disparity.
        pandas.testing.assert_series_equal(pd_result, bf_result, check_dtype=False)

        # Make sure the read_gbq_function path works for this function.
        featurize_ref = session.read_gbq_function(featurize.bigframes_bigquery_function)

        assert hasattr(featurize_ref, "bigframes_bigquery_function")
        assert not hasattr(featurize_ref, "bigframes_remote_function")
        assert (
            featurize_ref.bigframes_bigquery_function
            == featurize.bigframes_bigquery_function
        )

        # Test on the function from read_gbq_function.
        got = featurize_ref(10)
        assert got == [array_dtype(i) for i in [10, 11, 12]]

        bf_result_gbq = bf_int64_col.apply(featurize_ref).to_pandas()
        pandas.testing.assert_series_equal(bf_result_gbq, pd_result, check_dtype=False)

    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(featurize, session.bqclient)


@pytest.mark.parametrize(
    ("typ",),
    [
        pytest.param(int),
        pytest.param(float),
        pytest.param(bool),
        pytest.param(str),
        pytest.param(bytes),
    ],
)
def test_managed_function_series_apply(
    session,
    typ,
    scalars_dfs,
):
    try:

        @session.udf()
        def foo(x: int) -> typ:  # type:ignore
            # The bytes() constructor expects a non-negative interger as its arg.
            return typ(abs(x))

        # Function should still work normally.
        assert foo(-2) == typ(2)

        assert hasattr(foo, "bigframes_bigquery_function")
        assert hasattr(foo, "ibis_node")
        assert hasattr(foo, "input_dtypes")
        assert hasattr(foo, "output_dtype")
        assert hasattr(foo, "bigframes_bigquery_function_output_dtype")

        scalars_df, scalars_pandas_df = scalars_dfs

        bf_result_col = scalars_df["int64_too"].apply(foo)
        bf_result = (
            scalars_df["int64_too"].to_frame().assign(result=bf_result_col).to_pandas()
        )

        pd_result_col = scalars_pandas_df["int64_too"].apply(foo)
        pd_result = (
            scalars_pandas_df["int64_too"].to_frame().assign(result=pd_result_col)
        )

        pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)

        # Make sure the read_gbq_function path works for this function.
        foo_ref = session.read_gbq_function(
            function_name=foo.bigframes_bigquery_function,  # type: ignore
        )
        assert hasattr(foo_ref, "bigframes_bigquery_function")
        assert not hasattr(foo_ref, "bigframes_remote_function")
        assert foo.bigframes_bigquery_function == foo_ref.bigframes_bigquery_function  # type: ignore

        bf_result_col_gbq = scalars_df["int64_too"].apply(foo_ref)
        bf_result_gbq = (
            scalars_df["int64_too"]
            .to_frame()
            .assign(result=bf_result_col_gbq)
            .to_pandas()
        )

        pandas.testing.assert_frame_equal(bf_result_gbq, pd_result, check_dtype=False)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(foo, session.bqclient)


@pytest.mark.parametrize(
    ("typ",),
    [
        pytest.param(int),
        pytest.param(float),
        pytest.param(bool),
        pytest.param(str),
    ],
)
def test_managed_function_series_apply_array_output(
    session,
    typ,
    scalars_dfs,
):
    try:

        @session.udf()
        def foo_list(x: int) -> list[typ]:  # type:ignore
            # The bytes() constructor expects a non-negative interger as its arg.
            return [typ(abs(x)), typ(abs(x) + 1)]

        scalars_df, scalars_pandas_df = scalars_dfs

        bf_result_col = scalars_df["int64_too"].apply(foo_list)
        bf_result = (
            scalars_df["int64_too"].to_frame().assign(result=bf_result_col).to_pandas()
        )

        pd_result_col = scalars_pandas_df["int64_too"].apply(foo_list)
        pd_result = (
            scalars_pandas_df["int64_too"].to_frame().assign(result=pd_result_col)
        )

        # Ignore any dtype difference.
        pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(foo_list, session.bqclient)


def test_managed_function_series_combine(session, scalars_dfs):
    try:
        # This function is deliberately written to not work with NA input.
        def add(x: int, y: int) -> int:
            return x + y

        scalars_df, scalars_pandas_df = scalars_dfs
        int_col_name_with_nulls = "int64_col"
        int_col_name_no_nulls = "int64_too"
        bf_df = scalars_df[[int_col_name_with_nulls, int_col_name_no_nulls]]
        pd_df = scalars_pandas_df[[int_col_name_with_nulls, int_col_name_no_nulls]]

        # make sure there are NA values in the test column.
        assert any([pandas.isna(val) for val in bf_df[int_col_name_with_nulls]])

        add_managed_func = session.udf()(add)

        # with nulls in the series the managed function application would fail.
        with pytest.raises(
            google.api_core.exceptions.BadRequest, match="unsupported operand"
        ):
            bf_df[int_col_name_with_nulls].combine(
                bf_df[int_col_name_no_nulls], add_managed_func
            ).to_pandas()

        # after filtering out nulls the managed function application should work
        # similar to pandas.
        pd_filter = pd_df[int_col_name_with_nulls].notnull()
        pd_result = pd_df[pd_filter][int_col_name_with_nulls].combine(
            pd_df[pd_filter][int_col_name_no_nulls], add
        )
        bf_filter = bf_df[int_col_name_with_nulls].notnull()
        bf_result = (
            bf_df[bf_filter][int_col_name_with_nulls]
            .combine(bf_df[bf_filter][int_col_name_no_nulls], add_managed_func)
            .to_pandas()
        )

        # ignore any dtype difference.
        pandas.testing.assert_series_equal(pd_result, bf_result, check_dtype=False)

        # Make sure the read_gbq_function path works for this function.
        add_managed_func_ref = session.read_gbq_function(
            add_managed_func.bigframes_bigquery_function
        )
        bf_result = (
            bf_df[bf_filter][int_col_name_with_nulls]
            .combine(bf_df[bf_filter][int_col_name_no_nulls], add_managed_func_ref)
            .to_pandas()
        )
        pandas.testing.assert_series_equal(bf_result, pd_result, check_dtype=False)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(add_managed_func, session.bqclient)


def test_managed_function_series_combine_array_output(session, scalars_dfs):
    try:

        def add_list(x: int, y: int) -> list[int]:
            return [x, y]

        scalars_df, scalars_pandas_df = scalars_dfs
        int_col_name_with_nulls = "int64_col"
        int_col_name_no_nulls = "int64_too"
        bf_df = scalars_df[[int_col_name_with_nulls, int_col_name_no_nulls]]
        pd_df = scalars_pandas_df[[int_col_name_with_nulls, int_col_name_no_nulls]]

        # Make sure there are NA values in the test column.
        assert any([pandas.isna(val) for val in bf_df[int_col_name_with_nulls]])

        add_list_managed_func = session.udf()(add_list)

        # After filtering out nulls the managed function application should work
        # similar to pandas.
        pd_filter = pd_df[int_col_name_with_nulls].notnull()
        pd_result = pd_df[pd_filter][int_col_name_with_nulls].combine(
            pd_df[pd_filter][int_col_name_no_nulls], add_list
        )
        bf_filter = bf_df[int_col_name_with_nulls].notnull()
        bf_result = (
            bf_df[bf_filter][int_col_name_with_nulls]
            .combine(bf_df[bf_filter][int_col_name_no_nulls], add_list_managed_func)
            .to_pandas()
        )

        # Ignore any dtype difference.
        pandas.testing.assert_series_equal(pd_result, bf_result, check_dtype=False)

        # Make sure the read_gbq_function path works for this function.
        add_list_managed_func_ref = session.read_gbq_function(
            function_name=add_list_managed_func.bigframes_bigquery_function,  # type: ignore
        )

        assert hasattr(add_list_managed_func_ref, "bigframes_bigquery_function")
        assert not hasattr(add_list_managed_func_ref, "bigframes_remote_function")
        assert (
            add_list_managed_func_ref.bigframes_bigquery_function
            == add_list_managed_func.bigframes_bigquery_function
        )

        # Test on the function from read_gbq_function.
        got = add_list_managed_func_ref(10, 38)
        assert got == [10, 38]

        bf_result_gbq = (
            bf_df[bf_filter][int_col_name_with_nulls]
            .combine(bf_df[bf_filter][int_col_name_no_nulls], add_list_managed_func_ref)
            .to_pandas()
        )

        pandas.testing.assert_series_equal(bf_result_gbq, pd_result, check_dtype=False)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(add_list_managed_func, session.bqclient)


def test_managed_function_dataframe_map(session, scalars_dfs):
    try:

        def add_one(x):
            return x + 1

        mf_add_one = session.udf(
            input_types=[int],
            output_type=int,
        )(add_one)

        scalars_df, scalars_pandas_df = scalars_dfs
        int64_cols = ["int64_col", "int64_too"]

        bf_int64_df = scalars_df[int64_cols]
        bf_int64_df_filtered = bf_int64_df.dropna()
        bf_result = bf_int64_df_filtered.map(mf_add_one).to_pandas()

        pd_int64_df = scalars_pandas_df[int64_cols]
        pd_int64_df_filtered = pd_int64_df.dropna()
        pd_result = pd_int64_df_filtered.map(add_one)
        # TODO(shobs): Figure why pandas .map() changes the dtype, i.e.
        # pd_int64_df_filtered.dtype is Int64Dtype()
        # pd_int64_df_filtered.map(lambda x: x).dtype is int64.
        # For this test let's force the pandas dtype to be same as input.
        for col in pd_result:
            pd_result[col] = pd_result[col].astype(pd_int64_df_filtered[col].dtype)

        pandas.testing.assert_frame_equal(bf_result, pd_result)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(mf_add_one, session.bqclient)


def test_managed_function_dataframe_map_array_output(
    session, scalars_dfs, dataset_id_permanent
):
    try:

        def add_one_list(x):
            return [x + 1] * 3

        mf_add_one_list = session.udf(
            input_types=[int],
            output_type=list[int],
        )(add_one_list)

        scalars_df, scalars_pandas_df = scalars_dfs
        int64_cols = ["int64_col", "int64_too"]

        bf_int64_df = scalars_df[int64_cols]
        bf_int64_df_filtered = bf_int64_df.dropna()
        bf_result = bf_int64_df_filtered.map(mf_add_one_list).to_pandas()

        pd_int64_df = scalars_pandas_df[int64_cols]
        pd_int64_df_filtered = pd_int64_df.dropna()
        pd_result = pd_int64_df_filtered.map(add_one_list)

        # Ignore any dtype difference.
        pandas.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)

        # Make sure the read_gbq_function path works for this function.
        mf_add_one_list_ref = session.read_gbq_function(
            function_name=mf_add_one_list.bigframes_bigquery_function,  # type: ignore
        )

        bf_result_gbq = bf_int64_df_filtered.map(mf_add_one_list_ref).to_pandas()
        pandas.testing.assert_frame_equal(bf_result_gbq, pd_result, check_dtype=False)
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(mf_add_one_list, session.bqclient)


def test_managed_function_dataframe_apply_axis_1(session, scalars_dfs):
    try:
        scalars_df, scalars_pandas_df = scalars_dfs
        series = scalars_df["int64_too"]
        series_pandas = scalars_pandas_df["int64_too"]

        def add_ints(x, y):
            return x + y

        add_ints_mf = session.udf(
            input_types=[int, int],
            output_type=int,
        )(add_ints)
        assert add_ints_mf.bigframes_bigquery_function  # type: ignore

        with pytest.warns(
            bigframes.exceptions.PreviewWarning, match="axis=1 scenario is in preview."
        ):
            bf_result = (
                bpd.DataFrame({"x": series, "y": series})
                .apply(add_ints_mf, axis=1)
                .to_pandas()
            )

        pd_result = pandas.DataFrame({"x": series_pandas, "y": series_pandas}).apply(
            lambda row: add_ints(row["x"], row["y"]), axis=1
        )

        pandas.testing.assert_series_equal(
            pd_result, bf_result, check_dtype=False, check_exact=True
        )
    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(add_ints_mf, session.bqclient)


def test_managed_function_dataframe_apply_axis_1_array_output(session):
    bf_df = bigframes.dataframe.DataFrame(
        {
            "Id": [1, 2, 3],
            "Age": [22.5, 23, 23.5],
            "Name": ["alpha", "beta", "gamma"],
        }
    )

    expected_dtypes = (
        bigframes.dtypes.INT_DTYPE,
        bigframes.dtypes.FLOAT_DTYPE,
        bigframes.dtypes.STRING_DTYPE,
    )

    # Assert the dataframe dtypes.
    assert tuple(bf_df.dtypes) == expected_dtypes

    try:

        @session.udf(input_types=[int, float, str], output_type=list[str])
        def foo(x, y, z):
            return [str(x), str(y), z]

        assert getattr(foo, "is_row_processor") is False
        assert getattr(foo, "input_dtypes") == expected_dtypes
        assert getattr(foo, "output_dtype") == pandas.ArrowDtype(
            pyarrow.list_(
                bigframes.dtypes.bigframes_dtype_to_arrow_dtype(
                    bigframes.dtypes.STRING_DTYPE
                )
            )
        )
        assert getattr(foo, "output_dtype") == getattr(
            foo, "bigframes_bigquery_function_output_dtype"
        )

        # Fails to apply on dataframe with incompatible number of columns.
        with pytest.raises(
            ValueError,
            match="^BigFrames BigQuery function takes 3 arguments but DataFrame has 2 columns\\.$",
        ):
            bf_df[["Id", "Age"]].apply(foo, axis=1)

        with pytest.raises(
            ValueError,
            match="^BigFrames BigQuery function takes 3 arguments but DataFrame has 4 columns\\.$",
        ):
            bf_df.assign(Country="lalaland").apply(foo, axis=1)

        # Fails to apply on dataframe with incompatible column datatypes.
        with pytest.raises(
            ValueError,
            match="^BigFrames BigQuery function takes arguments of types .* but DataFrame dtypes are .*",
        ):
            bf_df.assign(Age=bf_df["Age"].astype("Int64")).apply(foo, axis=1)

        # Successfully applies to dataframe with matching number of columns.
        # and their datatypes.
        with pytest.warns(
            bigframes.exceptions.PreviewWarning,
            match="axis=1 scenario is in preview.",
        ):
            bf_result = bf_df.apply(foo, axis=1).to_pandas()

        # Since this scenario is not pandas-like, let's handcraft the
        # expected result.
        expected_result = pandas.Series(
            [
                ["1", "22.5", "alpha"],
                ["2", "23.0", "beta"],
                ["3", "23.5", "gamma"],
            ]
        )

        pandas.testing.assert_series_equal(
            expected_result, bf_result, check_dtype=False, check_index_type=False
        )

        # Make sure the read_gbq_function path works for this function.
        foo_ref = session.read_gbq_function(foo.bigframes_bigquery_function)

        assert hasattr(foo_ref, "bigframes_bigquery_function")
        assert not hasattr(foo_ref, "bigframes_remote_function")
        assert foo_ref.bigframes_bigquery_function == foo.bigframes_bigquery_function

        # Test on the function from read_gbq_function.
        got = foo_ref(10, 38, "hello")
        assert got == ["10", "38.0", "hello"]

        with pytest.warns(
            bigframes.exceptions.PreviewWarning,
            match="axis=1 scenario is in preview.",
        ):
            bf_result_gbq = bf_df.apply(foo_ref, axis=1).to_pandas()

        pandas.testing.assert_series_equal(
            bf_result_gbq, expected_result, check_dtype=False, check_index_type=False
        )

    finally:
        # Clean up the gcp assets created for the managed function.
        cleanup_function_assets(foo, session.bqclient)
