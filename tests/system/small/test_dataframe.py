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

import geopandas as gpd  # type: ignore
import numpy as np
import pandas as pd
import pandas.testing
import pyarrow as pa  # type: ignore
import pytest

from tests.system.utils import (
    assert_pandas_df_equal_ignore_ordering,
    assert_series_equal_ignoring_order,
)


def test_get_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name = "int64_col"
    series = scalars_df[col_name]
    bf_result = series.compute()
    pd_result = scalars_pandas_df[col_name]
    assert_series_equal_ignoring_order(bf_result, pd_result)


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
    assert_series_equal_ignoring_order(bf_result, pd_result)


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


def test_repr_w_all_rows(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    if scalars_pandas_df.index.name is None:
        # Note: Not quite the same as no index / default index, but hopefully
        # simulates it well enough while being consistent enough for string
        # comparison to work.
        scalars_df = scalars_df.set_index("rowindex", drop=False).sort_index()
        scalars_df.index.name = None

    # When there are 10 or fewer rows, the outputs should be identical.
    actual = repr(scalars_df.head(10))
    expected = repr(scalars_pandas_df.head(10))
    assert actual == expected


def test_df_column_name_with_space(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name_dict = {"bool_col": "bool  col"}
    df_pandas = scalars_df.rename(col_name_dict).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.rename(columns=col_name_dict).columns
    )


def test_df_column_name_duplicate(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name_dict = {"int64_too": "int64_col"}
    df_pandas = scalars_df.rename(col_name_dict).compute()
    pd.testing.assert_index_equal(
        df_pandas.columns, scalars_pandas_df.rename(columns=col_name_dict).columns
    )


def test_get_df_column_name_duplicate(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    col_name_dict = {"int64_too": "int64_col"}

    bf_result = scalars_df.rename(col_name_dict)["int64_col"].compute()
    pd_result = scalars_pandas_df.rename(columns=col_name_dict)["int64_col"]
    pd.testing.assert_index_equal(bf_result.columns, pd_result.columns)


def test_filter_df(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    bf_bool_series = scalars_df["bool_col"]
    bf_result = scalars_df[bf_bool_series].compute()

    pd_bool_series = scalars_pandas_df["bool_col"]
    pd_result = scalars_pandas_df[pd_bool_series]

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_assign_new_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"new_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    # Convert default pandas dtypes `int64` to match BigFrames dtypes.
    pd_result["new_col"] = pd_result["new_col"].astype("Int64")

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_assign_existing_column(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"int64_col": 2}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    # Convert default pandas dtypes `int64` to match BigFrames dtypes.
    pd_result["int64_col"] = pd_result["int64_col"].astype("Int64")

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_assign_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    column_name = "int64_col"
    df = scalars_df.assign(new_col=scalars_df[column_name])
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(new_col=scalars_pandas_df[column_name])

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_assign_sequential(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    kwargs = {"int64_col": 2, "new_col": 3, "new_col2": 4}
    df = scalars_df.assign(**kwargs)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.assign(**kwargs)

    # Convert default pandas dtypes `int64` to match BigFrames dtypes.
    pd_result["int64_col"] = pd_result["int64_col"].astype("Int64")
    pd_result["new_col"] = pd_result["new_col"].astype("Int64")
    pd_result["new_col2"] = pd_result["new_col2"].astype("Int64")

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


# Different table expression must have Index
def test_assign_different_df(
    scalars_df_index, scalars_df_2_index, scalars_pandas_df_index
):
    column_name = "int64_col"
    df = scalars_df_index.assign(new_col=scalars_df_2_index[column_name])
    bf_result = df.compute()
    # Doesn't matter to pandas if it comes from the same DF or a different DF.
    pd_result = scalars_pandas_df_index.assign(
        new_col=scalars_pandas_df_index[column_name]
    )

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_dropna(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    df = scalars_df.dropna()
    bf_result = df.compute()
    pd_result = scalars_pandas_df.dropna()

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


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

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.parametrize(
    ("merge_how",),
    [
        ("inner",),
        ("outer",),
        ("left",),
        ("right",),
    ],
)
def test_merge_custom_col_name(scalars_dfs, merge_how):
    scalars_df, scalars_pandas_df = scalars_dfs
    # Pandas join allows on NaN, well BQ excludes those rows.
    # TODO(garrettwu): Figure out how we want to deal with null values in joins.
    scalars_df = scalars_df.dropna()
    scalars_pandas_df = scalars_pandas_df.dropna()

    left_columns = ["int64_col", "float64_col"]
    right_columns = ["int64_col", "bool_col", "string_col"]
    on = "int64_col"
    rename_columns = {"float64_col": "f64_col"}

    left = scalars_df[left_columns]
    left = left.rename(columns=rename_columns)
    right = scalars_df[right_columns]
    df = left.merge(right, merge_how, on)
    bf_result = df.compute()

    pandas_left_df = scalars_pandas_df[left_columns]
    pandas_left_df = pandas_left_df.rename(columns=rename_columns)
    pandas_right_df = scalars_pandas_df[right_columns]
    pd_result = pandas_left_df.merge(pandas_right_df, merge_how, on)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_get_dtypes(scalars_df_default_index):
    dtypes = scalars_df_default_index.dtypes
    pd.testing.assert_series_equal(
        dtypes,
        pd.Series(
            {
                "bool_col": pd.BooleanDtype(),
                "bytes_col": np.dtype("O"),
                "date_col": pd.ArrowDtype(pa.date32()),
                "datetime_col": pd.ArrowDtype(pa.timestamp("us")),
                "geography_col": gpd.array.GeometryDtype(),
                "int64_col": pd.Int64Dtype(),
                "int64_too": pd.Int64Dtype(),
                "numeric_col": np.dtype("O"),
                "float64_col": pd.Float64Dtype(),
                "rowindex": pd.Int64Dtype(),
                "rowindex_2": pd.Int64Dtype(),
                "string_col": pd.StringDtype(storage="pyarrow"),
                "time_col": pd.ArrowDtype(pa.time64("us")),
                # TODO(chelsealin): should be pd.ArrowDtype(pa.timestamp("us", tz="UTC")
                # after fixing b/279503940.
                "timestamp_col": pd.ArrowDtype(pa.timestamp("us")),
            }
        ),
    )


def test_shape(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = scalars_df.shape
    pd_result = scalars_pandas_df.shape

    assert bf_result == pd_result


def test_size(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = scalars_df.size
    pd_result = scalars_pandas_df.size

    assert bf_result == pd_result


def test_ndim(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = scalars_df.ndim
    pd_result = scalars_pandas_df.ndim

    assert bf_result == pd_result


def test_empty_false(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    bf_result = scalars_df.empty
    pd_result = scalars_pandas_df.empty

    assert bf_result == pd_result


def test_empty_true(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    bf_result = scalars_df[[]].empty
    pd_result = scalars_pandas_df[[]].empty

    assert bf_result == pd_result


@pytest.mark.parametrize(
    ("drop",),
    ((True,), (False,)),
)
def test_reset_index(scalars_df_index, scalars_pandas_df_index, drop):
    df = scalars_df_index.reset_index(drop=drop)
    bf_result = df.compute()
    pd_result = scalars_pandas_df_index.reset_index(drop=drop)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.parametrize(
    ("drop",),
    (
        (True,),
        (False,),
    ),
)
@pytest.mark.parametrize(
    ("index_column",),
    (("int64_too",), ("string_col",), ("timestamp_col",)),
)
def test_set_index(scalars_dfs, index_column, drop):
    scalars_df, scalars_pandas_df = scalars_dfs
    df = scalars_df.set_index(index_column, drop=drop)
    bf_result = df.compute()
    pd_result = scalars_pandas_df.set_index(index_column, drop=drop)

    # Sort to disambiguate when there are duplicate index labels.
    # Note: Doesn't use assert_pandas_df_equal_ignore_ordering because we get
    # "ValueError: 'timestamp_col' is both an index level and a column label,
    # which is ambiguous" when trying to sort by a column with the same name as
    # the index.
    bf_result = bf_result.sort_values("rowindex_2")
    pd_result = pd_result.sort_values("rowindex_2")

    pandas.testing.assert_frame_equal(bf_result, pd_result)


def test_df_abs(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    columns = ["int64_col", "int64_too", "float64_col"]

    bf_result = scalars_df[columns].abs().compute()
    pd_result = scalars_pandas_df[columns].abs()

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_df_isnull(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    columns = ["int64_col", "int64_too", "string_col", "bool_col"]
    bf_result = scalars_df[columns].isnull().compute()
    pd_result = scalars_pandas_df[columns].isnull()

    # One of dtype mismatches to be documented. Here, the `bf_result.dtype` is
    # `BooleanDtype` but the `pd_result.dtype` is `bool`.
    pd_result["int64_col"] = pd_result["int64_col"].astype(pd.BooleanDtype())
    pd_result["int64_too"] = pd_result["int64_too"].astype(pd.BooleanDtype())
    pd_result["string_col"] = pd_result["string_col"].astype(pd.BooleanDtype())
    pd_result["bool_col"] = pd_result["bool_col"].astype(pd.BooleanDtype())

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_df_notnull(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    columns = ["int64_col", "int64_too", "string_col", "bool_col"]
    bf_result = scalars_df[columns].notnull().compute()
    pd_result = scalars_pandas_df[columns].notnull()

    # One of dtype mismatches to be documented. Here, the `bf_result.dtype` is
    # `BooleanDtype` but the `pd_result.dtype` is `bool`.
    pd_result["int64_col"] = pd_result["int64_col"].astype(pd.BooleanDtype())
    pd_result["int64_too"] = pd_result["int64_too"].astype(pd.BooleanDtype())
    pd_result["string_col"] = pd_result["string_col"].astype(pd.BooleanDtype())
    pd_result["bool_col"] = pd_result["bool_col"].astype(pd.BooleanDtype())

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@pytest.mark.parametrize(
    ("operator"),
    [
        (lambda x, y: x + y),
        (lambda x, y: x - y),
        (lambda x, y: x * y),
        # TODO(garrettwu): check why dtypes of floor divide are different as pandas.
        # (lambda x, y: x / y),
        # (lambda x, y: x // y),
    ],
    ids=[
        "add",
        "subtract",
        "multiply",
        # "true_divide",
        # "floor_divide",
    ],
)
# TODO(garrettwu): deal with NA values and 0 divisions
@pytest.mark.parametrize(("other_scalar"), [1, 2.5])
@pytest.mark.parametrize(("reverse_operands"), [True, False])
def test_scalar_bi_op(scalars_dfs, operator, other_scalar, reverse_operands):
    scalars_df, scalars_pandas_df = scalars_dfs
    columns = ["int64_col", "float64_col"]

    maybe_reversed_op = (lambda x, y: operator(y, x)) if reverse_operands else operator

    bf_result = maybe_reversed_op(scalars_df[columns], other_scalar).compute()
    pd_result = maybe_reversed_op(scalars_pandas_df[columns], other_scalar)

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_scalar_bi_op_str_exception(scalars_dfs):
    scalars_df, _ = scalars_dfs
    columns = ["string_col"]
    with pytest.raises(TypeError):
        (scalars_df[columns] + 1).compute()


all_joins = pytest.mark.parametrize(
    ("how",),
    (
        ("outer",),
        ("left",),
        ("right",),
        ("inner",),
    ),
)


@all_joins
def test_join_same_table(scalars_dfs, how):
    bf_df, pd_df = scalars_dfs
    if how == "right" and pd_df.index.name != "rowindex":
        pytest.skip("right join not supported without an index")

    bf_df_a = bf_df[["string_col", "int64_col"]]
    bf_df_b = bf_df[["float64_col"]]
    bf_result = bf_df_a.join(bf_df_b, how=how).compute()
    pd_df_a = pd_df[["string_col", "int64_col"]]
    pd_df_b = pd_df[["float64_col"]]
    pd_result = pd_df_a.join(pd_df_b, how=how)
    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


@all_joins
def test_join_different_table(
    scalars_df_index, scalars_df_2_index, scalars_pandas_df_index, how
):
    bf_df_a = scalars_df_index[["string_col", "int64_col"]]
    bf_df_b = scalars_df_2_index.dropna()[["float64_col"]]
    bf_result = bf_df_a.join(bf_df_b, how=how).compute()
    pd_df_a = scalars_pandas_df_index[["string_col", "int64_col"]]
    pd_df_b = scalars_pandas_df_index.dropna()[["float64_col"]]
    pd_result = pd_df_a.join(pd_df_b, how=how)
    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_join_duplicate_columns_raises_not_implemented(scalars_dfs):
    scalars_df, _ = scalars_dfs
    df_a = scalars_df[["string_col", "float64_col"]]
    df_b = scalars_df[["float64_col"]]
    with pytest.raises(NotImplementedError):
        df_a.join(df_b, how="outer").compute()
