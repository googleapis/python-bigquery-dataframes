import db_dtypes  # type: ignore
import numpy as np
import pandas as pd
import pytest


def test_get_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    series = scalars_df[col_name]
    bf_result = series.compute()
    pd_result = scalars_pandas_df[col_name]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(bf_result, pd_result)


def test_hasattr(scalars_dfs):
    scalars_df, _ = scalars_dfs
    assert hasattr(scalars_df, "int64_col")
    assert hasattr(scalars_df, "head")
    assert not hasattr(scalars_df, "not_exist")


def test_get_column_by_attr(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    series = scalars_df.int64_col
    bf_result = series.compute()
    pd_result = scalars_pandas_df.int64_col

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(bf_result, pd_result)


def test_get_columns(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_names = ["bool_col", "float64_col", "int64_col"]
    df_subset = scalars_df[col_names]
    df_pandas = df_subset.compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df[col_names].columns
    )


def test_drop_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    df_pandas = scalars_df.drop(col_name).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_name).columns
    )


def test_drop_columns(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_names = ["int64_col", "geography_col", "time_col"]
    df_pandas = scalars_df.drop(col_names).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.drop(columns=col_names).columns
    )


def test_rename(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name_dict = {"bool_col": "boolean_col"}
    df_pandas = scalars_df.rename(col_name_dict).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.rename(columns=col_name_dict).columns
    )


def test_filter_df(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    bf_bool_series = scalars_df["bool_col"]
    bf_result = scalars_df[bf_bool_series].compute()

    pd_bool_series = scalars_pandas_df["bool_col"]
    pd_result = scalars_pandas_df[pd_bool_series]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_assign_new_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"new_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    # dtype discrepencies of Int64(ibis) vs int64(pandas)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_assign_existing_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"int64_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    # dtype discrepencies of Int64(ibis) vs int64(pandas)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_assign_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    column_name = "int64_col"
    df = scalars_df.assign(new_col=scalars_df[column_name])
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(new_col=scalars_pandas_df[column_name])

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    # dtype discrepencies of Int64(ibis) vs int64(pandas)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_dropna(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    df = scalars_df.dropna()
    bf_result = df.compute()
    pd_result = scalars_pandas_df.dropna()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values("rowindex", ignore_index=True)
        pd_result = pd_result.sort_values("rowindex", ignore_index=True)

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    ("merge_how",),
    [
        ("inner",),
        ("outer",),
        ("left",),
        ("right",),
    ],
)
def test_merge(scalars_dfs, merge_how):
    scalars_df, scalars_pandas_df = scalars_dfs
    # Pandas join allows on NaN, well BQ excludes those rows.
    # TODO(garrettwu): Figure out how we want to deal with null values in joins.
    scalars_df = scalars_df.dropna()
    scalars_pandas_df = scalars_pandas_df.dropna()

    left_columns = ["int64_col", "float64_col"]
    right_columns = ["int64_col", "bool_col", "string_col"]
    on = "int64_col"

    left = scalars_df[left_columns]
    right = scalars_df[right_columns]
    df = left.merge(right, merge_how, on)
    bf_result = df.compute()

    pd_result = scalars_pandas_df[left_columns].merge(
        scalars_pandas_df[right_columns], merge_how, on
    )

    # Sort by a column to get consistent results.
    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(on, ignore_index=True)
        pd_result = pd_result.sort_values(on, ignore_index=True)
    # TODO(garrettwu): enable check_type once BF type issue is solved.
    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


def test_get_dtypes(scalars_df_no_index):
    dtypes = scalars_df_no_index.dtypes
    pd.testing.assert_series_equal(
        dtypes,
        pd.Series(
            {
                "bool_col": pd.BooleanDtype(),
                "bytes_col": np.dtype("O"),
                "date_col": db_dtypes.DateDtype(),
                "datetime_col": np.dtype("datetime64[us]"),
                "geography_col": np.dtype("O"),
                "int64_col": pd.Int64Dtype(),
                "int64_too": pd.Int64Dtype(),
                "numeric_col": np.dtype("O"),
                "float64_col": pd.Float64Dtype(),
                "rowindex": pd.Int64Dtype(),
                "rowindex_2": pd.Int64Dtype(),
                "string_col": pd.StringDtype(),
                "time_col": db_dtypes.TimeDtype(),
                # TODO(bmil): should be:
                # "timestamp_col": pd.DatetimeTZDtype(unit="us", tz="UTC")}))
                "timestamp_col": np.dtype("datetime64[us]"),
            }
        ),
    )


@pytest.mark.parametrize(
    ("drop",),
    ((True,), (False,)),
)
def test_reset_index(scalars_df_index, scalars_pandas_df_index, drop):
    df = scalars_df_index.reset_index(drop=drop)
    bf_result = df.compute()
    pd_result = scalars_pandas_df_index.reset_index(drop=drop)

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )


@pytest.mark.parametrize(
    ("index_column",),
    (("int64_too",), ("string_col",), ("timestamp_col",)),
)
def test_set_index(scalars_dfs, index_column):
    scalars_df, scalars_pandas_df = scalars_dfs
    df = scalars_df.set_index(index_column)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.set_index(index_column)

    # Sort to disambiguate when there are duplicate index labels.
    bf_result = bf_result.sort_values("rowindex_2")
    pd_result = pd_result.sort_values("rowindex_2")

    pd.testing.assert_frame_equal(
        bf_result,
        pd_result,
        check_column_type=False,
        check_dtype=False,
        check_index_type=False,
    )
