import numpy
import pandas as pd
import pytest


def assert_float_close(left, right, allowed=1.0e-5):
    assert abs(left - right) < allowed


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
        ("int64_too", numpy.dtype("int64")),
        # TODO(swast): Use a more efficient type.
        ("string_col", numpy.dtype("object")),
        # TODO(swast): Update ibis-bigquery backend to use
        # db_dtypes.TimeDtype() when available.
        ("time_col", numpy.dtype("object")),
        # TODO(swast): Make sure timestamps are associated with UTC timezone.
        ("timestamp_col", numpy.dtype("datetime64[ns]")),
    ],
)
def test_get_column(scalars_dfs, col_name, expected_dtype):
    scalars_df, scalars_pandas_df = scalars_dfs
    series = scalars_df[col_name]
    series_pandas = series.compute()
    assert series_pandas.dtype == expected_dtype
    assert series_pandas.shape[0] == scalars_pandas_df.shape[0]


@pytest.mark.parametrize(
    ("col_name",),
    (
        ("float64_col",),
        ("int64_too",),
    ),
)
def test_abs(scalars_dfs, col_name):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = scalars_df[col_name].abs().compute()
    pd_result = scalars_pandas_df[col_name].abs()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_find(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name].find("W").compute()
    pd_result = scalars_pandas_df[col_name].str.find("W")

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
    )


def test_len(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name].len().compute()
    pd_result = scalars_pandas_df[col_name].str.len()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_lower(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name].lower().compute()
    pd_result = scalars_pandas_df[col_name].str.lower()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_upper(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name].upper().compute()
    pd_result = scalars_pandas_df[col_name].str.upper()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


@pytest.mark.parametrize(
    ("other",),
    [
        (3,),
        (-6.2,),
    ],
)
def test_series_add_scalar(scalars_dfs, other):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = (scalars_df["float64_col"] + other).compute()
    pd_result = scalars_pandas_df["float64_col"] + other

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


@pytest.mark.parametrize(
    ("left_col", "right_col"),
    [
        ("float64_col", "float64_col"),
        ("int64_col", "float64_col"),
        ("int64_col", "int64_col"),
    ],
)
def test_series_add_bigframes_series(scalars_dfs, left_col, right_col):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = (scalars_df[left_col] + scalars_df[right_col]).compute()
    pd_result = scalars_pandas_df[left_col] + scalars_pandas_df[right_col]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result, check_names=False)


def test_series_add_different_table_no_index_value_error(
    scalars_df_no_index, scalars_df_2_no_index
):
    with pytest.raises(ValueError, match="scalars_too"):
        # Error should happen right away, not just at compute() time.
        scalars_df_no_index["float64_col"] + scalars_df_2_no_index["float64_col"]


def test_series_add_different_table_with_index(
    scalars_df_index, scalars_df_2_index, scalars_pandas_df_index
):
    scalars_pandas_df = scalars_pandas_df_index
    bf_result = scalars_df_index["float64_col"] + scalars_df_2_index["int64_col"]
    # When index values are unique, we can emulate with values from the same
    # DataFrame.
    pd_result = scalars_pandas_df["float64_col"] + scalars_pandas_df["int64_col"]
    pd_result.name = "float64_col"
    pd.testing.assert_series_equal(bf_result.compute(), pd_result)


def test_series_add_pandas_series_not_implemented(scalars_dfs):
    scalars_df, _ = scalars_dfs
    with pytest.raises(TypeError):
        (
            scalars_df["float64_col"]
            + pd.Series(
                [1, 1, 1, 1],
            )
        ).compute()


def test_reverse(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name].reverse().compute()
    pd_result = scalars_pandas_df[col_name].map(lambda x: x[::-1] if x else x)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_round(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "float64_col"
    bf_result = scalars_df[col_name].round().compute()
    pd_result = scalars_pandas_df[col_name].round()

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_eq_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_too"
    bf_result = scalars_df[col_name].eq(0).compute()
    pd_result = scalars_pandas_df[col_name].eq(0)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_eq_wider_type_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_too"
    bf_result = scalars_df[col_name].eq(1.0).compute()
    pd_result = scalars_pandas_df[col_name].eq(1.0)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_ne_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_too"
    bf_result = (scalars_df[col_name] != 0).compute()
    pd_result = scalars_pandas_df[col_name] != 0

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_eq_int_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_too"
    bf_result = (scalars_df[col_name] == 0).compute()
    pd_result = scalars_pandas_df[col_name] == 0

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


@pytest.mark.parametrize(
    ("col_name",),
    (
        ("string_col",),
        ("float64_col",),
        ("int64_too",),
    ),
)
def test_eq_same_type_series(scalars_dfs, col_name):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = (scalars_df[col_name] == scalars_df[col_name]).compute()
    pd_result = scalars_pandas_df[col_name] == scalars_pandas_df[col_name]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_ne_obj_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = (scalars_df[col_name] != scalars_df[col_name]).compute()
    pd_result = scalars_pandas_df[col_name] != scalars_pandas_df[col_name]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_indexing_using_unselected_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name][scalars_df["int64_too"].eq(0)].compute()
    pd_result = scalars_pandas_df[col_name][scalars_pandas_df["int64_too"].eq(0)]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_indexing_using_selected_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_result = scalars_df[col_name][
        scalars_df["string_col"].eq("Hello, World!")
    ].compute()
    pd_result = scalars_pandas_df[col_name][
        scalars_pandas_df["string_col"].eq("Hello, World!")
    ]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_nested_filter(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    string_col = scalars_df["string_col"]
    int64_too = scalars_df["int64_too"]
    bool_col = scalars_df["bool_col"]
    bf_result = string_col[int64_too == 0][~bool_col].compute()

    pd_string_col = scalars_pandas_df["string_col"]
    pd_int64_too = scalars_pandas_df["int64_too"]
    pd_bool_col = scalars_pandas_df["bool_col"] == bool(
        True
    )  # Convert from nullable bool to nonnullable bool usable as indexer
    pd_result = pd_string_col[pd_int64_too == 0][~pd_bool_col]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(pd_result, bf_result)


def test_binop_different_predicates(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    int64_col1 = scalars_df["int64_col"]
    int64_col2 = scalars_df["int64_col"]
    bool_col = scalars_df["bool_col"] == bool(True)  # Convert null to False
    bf_result = (int64_col1[bool_col] + int64_col2[bool_col.__invert__()]).compute()

    pd_int64_col1 = scalars_pandas_df["int64_col"]
    pd_int64_col2 = scalars_pandas_df["int64_col"]
    pd_bool_col = scalars_pandas_df["bool_col"].fillna(False)
    pd_result = pd_int64_col1[pd_bool_col] + pd_int64_col2[pd_bool_col.__invert__()]

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
        check_names=False,
    )


def test_binop_different_predicates2(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    int64_col1 = scalars_df["int64_col"]
    float64_col = scalars_df["float64_col"]
    bool_col = scalars_df["bool_col"] == bool(True)  # Convert null to False
    bf_result = (int64_col1[bool_col.__eq__(True)] + float64_col).compute()

    pd_int64_col1 = scalars_pandas_df["int64_col"]
    pd_float64_col = scalars_pandas_df["float64_col"]
    pd_bool_col = scalars_pandas_df["bool_col"].fillna(False)
    pd_result = pd_int64_col1[pd_bool_col] + pd_float64_col

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
        check_names=False,
    )


def test_mean(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    bf_result = scalars_df[col_name].mean().compute()
    pd_result = scalars_pandas_df[col_name].mean()
    assert_float_close(pd_result, bf_result)


def test_repr(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    if scalars_pandas_df.index.name != "rowindex":
        pytest.skip("Require index & ordering for consistent repr.")

    col_name = "string_col"
    bf_series = scalars_df[col_name]
    pd_series = scalars_pandas_df[col_name]
    assert repr(bf_series) == repr(pd_series)


def test_sum(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    bf_result = scalars_df[col_name].sum().compute()
    pd_result = scalars_pandas_df[col_name].sum()
    assert pd_result == bf_result


def test_groupby_sum(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    bf_series = scalars_df[col_name].groupby(scalars_df["int64_too"]).sum()
    # TODO(swast): Type cast should be unnecessary when we use nullable dtypes
    # everywhere.
    pd_series = (
        scalars_pandas_df[col_name]
        .astype(pd.Int64Dtype())
        .groupby(scalars_pandas_df["int64_too"])
        .sum()
    )
    # TODO(swast): Update groupby to use index based on group by key(s).
    # TODO(swast): BigFrames groupby should result in the same names as pandas.
    # e.g. int64_col_sum
    pd.testing.assert_series_equal(
        pd_series.sort_index(),
        bf_series.compute().astype(pd.Int64Dtype()).sort_index(),
        check_exact=False,
        check_names=False,
    )


def test_groupby_mean(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    bf_series = scalars_df[col_name].groupby(scalars_df["int64_too"]).mean()
    pd_series = (
        scalars_pandas_df[col_name].groupby(scalars_pandas_df["int64_too"]).mean()
    )
    # TODO(swast): Update groupby to use index based on group by key(s).
    # TODO(swast): BigFrames groupby should result in the same names as pandas.
    # e.g. int64_col_mean
    pd.testing.assert_series_equal(
        pd_series.sort_index(),
        bf_series.compute().sort_index(),
        check_exact=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    ["start", "stop"], [(0, 1), (3, 5), (100, 101), (None, 1), (0, 12), (0, None)]
)
def test_slice(scalars_dfs, start, stop):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "string_col"
    bf_series = scalars_df[col_name]
    bf_result = bf_series.slice(start, stop).compute()
    pd_series = scalars_pandas_df[col_name]
    pd_result = pd_series.str.slice(start, stop)

    if pd_result.index.name != "rowindex":
        bf_result = bf_result.sort_values(ignore_index=True)
        pd_result = pd_result.sort_values(ignore_index=True)

    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
    )
