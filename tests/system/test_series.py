import numpy
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ["col_name", "expected_dtype"],
    [
        # TODO(swast): Use pandas.BooleanDtype() to represent nullable bool.
        ("bool_col", numpy.dtype("object")),
        # TODO(swast): Use a more efficient type.
        ("bytes_col", numpy.dtype("object")),
        # TODO(swast): Update ibis-bigquery backend to use
        # db_dtypes.DateDtype() when available.
        ("date_col", numpy.dtype("datetime64[ns]")),
        ("datetime_col", numpy.dtype("datetime64[ns]")),
        ("float64_col", numpy.dtype("float64")),
        ("geography_col", numpy.dtype("object")),
        # TODO(swast): Don't accidentally discard data if NULL is present by
        # casting to float.
        ("int64_col", numpy.dtype("float64")),
        # TODO(swast): Use a more efficient type.
        ("numeric_col", numpy.dtype("object")),
        # TODO(swast): Use a consistent dtype for integers whether NULL is
        # present or not.
        ("rowindex", numpy.dtype("int64")),
        # TODO(swast): Use a more efficient type.
        ("string_col", numpy.dtype("object")),
        # TODO(swast): Update ibis-bigquery backend to use
        # db_dtypes.TimeDtype() when available.
        ("time_col", numpy.dtype("object")),
        # TODO(swast): Make sure timestamps are associated with UTC timezone.
        ("timestamp_col", numpy.dtype("datetime64[ns]")),
    ],
)
def test_get_column(scalars_df, scalars_pandas_df, col_name, expected_dtype):
    series = scalars_df[col_name]
    series_pandas = series.compute()
    assert series_pandas.dtype == expected_dtype
    assert series_pandas.shape[0] == scalars_pandas_df.shape[0]


def test_abs_floats(scalars_df, scalars_pandas_df):
    col_name = "float64_col"
    bf_series = scalars_df[col_name].abs()
    pandas_series = scalars_pandas_df[col_name].abs()
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


def test_abs_ints(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    bf_series = scalars_df[col_name].abs()
    pandas_series = scalars_pandas_df[col_name].abs()
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


def test_len(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name].len()
    pandas_series = scalars_pandas_df[col_name].str.len()
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


def test_lower(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name].lower()
    pandas_series = scalars_pandas_df[col_name].str.lower()
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


def test_upper(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name].upper()
    pandas_series = scalars_pandas_df[col_name].str.upper()
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


@pytest.mark.parametrize(
    ("other",),
    [
        (3,),
        (-6.2,),
    ],
)
def test_series_add_scalar(scalars_df, scalars_pandas_df, other):
    bf_series = scalars_df["float64_col"] + other
    pandas_series = scalars_pandas_df["float64_col"] + other
    pd.testing.assert_series_equal(pandas_series, bf_series.compute())


@pytest.mark.parametrize(
    ("left_col", "right_col"),
    [
        ("float64_col", "float64_col"),
        ("int64_col", "float64_col"),
        ("int64_col", "int64_col"),
    ],
)
def test_series_add_bigframes_series(
    scalars_df, scalars_pandas_df, left_col, right_col
):
    bf_series = scalars_df[left_col] + scalars_df[right_col]
    pandas_series = scalars_pandas_df[left_col] + scalars_pandas_df[right_col]
    pd.testing.assert_series_equal(
        pandas_series, bf_series.compute(), check_names=False
    )


def test_series_add_different_table_no_index_value_error(scalars_df, scalars_df_2):
    with pytest.raises(ValueError, match="scalars_too"):
        # Error should happen right away, not just at compute() time.
        scalars_df["float64_col"] + scalars_df_2["float64_col"]


def test_series_add_pandas_series_not_implemented(scalars_df):
    with pytest.raises(TypeError):
        (
            scalars_df["float64_col"]
            + pd.Series(
                [1, 1, 1, 1],
            )
        ).compute()


def test_reverse(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_result = scalars_df[col_name].reverse()
    pd_result = scalars_pandas_df[col_name].map(lambda x: x[::-1] if x else x)
    pd.testing.assert_series_equal(pd_result, bf_result.compute())


def test_round(scalars_df, scalars_pandas_df):
    col_name = "float64_col"
    bf_series = scalars_df[col_name].round()
    pd_series = scalars_pandas_df[col_name].round()
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_eq_scalar(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    bf_series = scalars_df[col_name].eq(0)
    pd_series = scalars_pandas_df[col_name].eq(0)
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_eq_wider_type_scalar(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    bf_series = scalars_df[col_name].eq(1.0)
    pd_series = scalars_pandas_df[col_name].eq(1.0)
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_ne_scalar(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    bf_series = scalars_df[col_name] != 0
    pd_series = scalars_pandas_df[col_name] != 0
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_eq_int_scalar(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    bf_series = scalars_df[col_name] == 0
    pd_series = scalars_pandas_df[col_name] == 0
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_eq_obj_series(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name] == scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name] == scalars_pandas_df[col_name]
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_ne_obj_series(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name] != scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name] != scalars_pandas_df[col_name]
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_eq_float_series(scalars_df, scalars_pandas_df):
    col_name = "float64_col"
    bf_series = scalars_df[col_name] == scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name] == scalars_pandas_df[col_name]
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_indexing_using_unselected_series(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name][scalars_df["rowindex"].eq(0)]
    pd_series = scalars_pandas_df[col_name][scalars_pandas_df["rowindex"].eq(0)]
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_indexing_using_selected_series(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name][scalars_df["string_col"].eq("Hello, World!")]
    pd_series = scalars_pandas_df[col_name][
        scalars_pandas_df["string_col"].eq("Hello, World!")
    ]
    pd.testing.assert_series_equal(pd_series, bf_series.compute())


def test_nested_filter(scalars_df, scalars_pandas_df):
    string_col = scalars_df["string_col"]
    rowindex = scalars_df["rowindex"]
    bool_col = scalars_df["bool_col"]
    bf_result = string_col[rowindex == 0][~bool_col]

    pd_string_col = scalars_pandas_df["string_col"]
    pd_rowindex = scalars_pandas_df["rowindex"]
    pd_bool_col = scalars_pandas_df["bool_col"] == bool(
        True
    )  # Convert from nullable bool to nonnullable bool usable as indexer
    pd_result = pd_string_col[pd_rowindex == 0][~pd_bool_col]

    pd.testing.assert_series_equal(pd_result, bf_result.compute(), check_index=False)


def test_mean(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    bf_result = scalars_df[col_name].mean().compute()
    pd_result = scalars_pandas_df[col_name].mean()
    assert pd_result == bf_result


def test_repr(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name]
    assert repr(bf_series) == repr(pd_series)


def test_sum(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    bf_result = scalars_df[col_name].sum().compute()
    pd_result = scalars_pandas_df[col_name].sum()
    assert pd_result == bf_result


def test_groupby_sum(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    bf_series = scalars_df[col_name].groupby(scalars_df["rowindex"]).sum()
    # TODO(swast): Type cast should be unnecessary when we use nullable dtypes
    # everywhere.
    pd_series = (
        scalars_pandas_df[col_name]
        .astype(pd.Int64Dtype())
        .groupby(scalars_pandas_df["rowindex"])
        .sum()
    )
    # TODO(swast): Update groupby to use index based on group by key(s).
    # TODO(swast): BigFrames groupby should result in the same names as pandas.
    # e.g. int64_col_sum
    pd.testing.assert_series_equal(
        pd_series,
        bf_series.compute().astype(pd.Int64Dtype()),
        check_index=False,
        check_names=False,
    )


def test_groupby_mean(scalars_df, scalars_pandas_df):
    col_name = "int64_col"
    bf_series = scalars_df[col_name].groupby(scalars_df["rowindex"]).mean()
    pd_series = (
        scalars_pandas_df[col_name].groupby(scalars_pandas_df["rowindex"]).mean()
    )
    # TODO(swast): Update groupby to use index based on group by key(s).
    # TODO(swast): BigFrames groupby should result in the same names as pandas.
    # e.g. int64_col_mean
    pd.testing.assert_series_equal(
        pd_series, bf_series.compute(), check_index=False, check_names=False
    )


@pytest.mark.parametrize(
    ["start", "stop"], [(0, 1), (3, 5), (100, 101), (None, 1), (0, 12), (0, None)]
)
def test_slice(scalars_df, scalars_pandas_df, start, stop):
    col_name = "string_col"
    bf_series = scalars_df[col_name]
    bf_result = bf_series.slice(start, stop).compute()
    pd_series = scalars_pandas_df[col_name]
    pd_result = pd_series.str.slice(start, stop)
    pd.testing.assert_series_equal(bf_result, pd_result)


def test_find(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name].find("W")
    pd_series = scalars_pandas_df[col_name].str.find("W")
    pd.testing.assert_series_equal(pd_series, bf_series.compute())
