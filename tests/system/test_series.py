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
    series = scalars_df[col_name]
    series_pandas = series.abs().compute()
    pd.testing.assert_series_equal(series_pandas, scalars_pandas_df[col_name].abs())


def test_abs_ints(scalars_df, scalars_pandas_df):
    col_name = "rowindex"
    series = scalars_df[col_name]
    series_pandas = series.abs().compute()
    pd.testing.assert_series_equal(series_pandas, scalars_pandas_df[col_name].abs())


def test_len(scalars_df):
    col_name = "string_col"
    series = scalars_df[col_name]
    series_pandas = series.len().compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([13, 5, None], name=col_name)
    )


def test_lower(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    series = scalars_df[col_name]
    series_pandas = series.lower().compute()
    pd.testing.assert_series_equal(
        series_pandas, scalars_pandas_df[col_name].str.lower()
    )


def test_upper(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    series = scalars_df[col_name]
    series_pandas = series.upper().compute()
    pd.testing.assert_series_equal(
        series_pandas, scalars_pandas_df[col_name].str.upper()
    )


@pytest.mark.parametrize(
    (
        "other",
        "expected",
    ),
    [
        (
            3,
            pd.Series([4.25, 5.5, numpy.nan], name="float64_col"),
        ),
        (
            -6.2,
            pd.Series([-4.95, -3.7, numpy.nan], name="float64_col"),
        ),
    ],
)
def test_series_add_scalar(scalars_df, other, expected):
    pd.testing.assert_series_equal(
        (scalars_df["float64_col"] + other).compute(), expected
    )


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
    bf_result = (scalars_df[left_col] + scalars_df[right_col]).compute()
    pd_result = scalars_pandas_df[left_col] + scalars_pandas_df[right_col]
    pd_result.name = left_col
    pd.testing.assert_series_equal(bf_result, pd_result)


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


def test_reverse(scalars_df):
    col_name = "string_col"
    series = scalars_df[col_name]
    series_pandas = series.reverse().compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series(["!dlroW ,olleH", "はちにんこ", None], name=col_name)
    )


def test_round(scalars_df):
    col_name = "float64_col"
    series = scalars_df[col_name]
    series_pandas = series.round().compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([1, 3, None], name=col_name)
    )


def test_eq_scalar(scalars_df):
    col_name = "rowindex"
    series_pandas = scalars_df[col_name].eq(0).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([True, True, False], name=col_name)
    )


def test_eq_wider_type_scalar(scalars_df):
    col_name = "rowindex"
    series_pandas = scalars_df[col_name].eq(1.0).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([False, False, True], name=col_name)
    )


def test_ne_scalar(scalars_df):
    col_name = "rowindex"
    series_pandas = (scalars_df[col_name] != 0).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([False, False, True], name=col_name)
    )


def test_eq_int_scalar(scalars_df):
    col_name = "rowindex"
    series_pandas = (0 == scalars_df[col_name]).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([True, True, False], name=col_name)
    )


def test_eq_obj_series(scalars_df):
    col_name = "string_col"
    series_pandas = (scalars_df[col_name] == scalars_df[col_name]).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([True, True, False], name=col_name)
    )


def test_ne_obj_series(scalars_df):
    col_name = "string_col"
    series_pandas = (scalars_df[col_name] != scalars_df[col_name]).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([False, False, True], name=col_name)
    )


def test_eq_float_series(scalars_df):
    col_name = "float64_col"
    series_pandas = (scalars_df[col_name] == scalars_df[col_name]).compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([True, True, False], name=col_name)
    )


def test_indexing_using_unselected_series(scalars_df):
    col_name = "string_col"
    series_pandas = scalars_df[col_name][scalars_df["rowindex"].eq(0)].compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series(["Hello, World!", "こんにちは"], name=col_name)
    )


def test_indexing_using_selected_series(scalars_df):
    col_name = "string_col"
    series_pandas = scalars_df[col_name][
        scalars_df["string_col"].eq("Hello, World!")
    ].compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series(["Hello, World!"], name=col_name)
    )


def test_nested_filter(scalars_df):
    string_col = scalars_df["string_col"]
    rowindex = scalars_df["rowindex"]
    bool_col = scalars_df["bool_col"]
    series_pandas = string_col[rowindex == 0][~bool_col].compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series(["こんにちは"], name="string_col")
    )


def test_mean(scalars_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    pandas_scalar = series.mean().compute()
    assert pandas_scalar == -432098766


def test_repr(scalars_df, scalars_pandas_df):
    col_name = "string_col"
    bf_series = scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name]
    assert repr(bf_series) == repr(pd_series)


def test_sum(scalars_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    pandas_scalar = series.sum().compute()
    assert pandas_scalar == -864197532


def test_groupby_sum(scalars_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    series_pandas = series.groupby(scalars_df["rowindex"]).sum().compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([-864197532.0, numpy.nan], name="int64_col_sum")
    )


def test_groupby_mean(scalars_df):
    col_name = "int64_col"
    series = scalars_df[col_name]
    series_pandas = series.groupby(scalars_df["rowindex"]).mean().compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([-432098766.0, numpy.nan], name="int64_col_mean")
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


def test_find(scalars_df):
    col_name = "string_col"
    series = scalars_df[col_name]
    series_pandas = series.find("W").compute()
    pd.testing.assert_series_equal(
        series_pandas, pd.Series([7, -1, None], name=col_name)
    )
