# Copyright 2025 Google LLC
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

import pandas as pd
import pytest

from bigframes.testing.utils import assert_pandas_df_equal


def test_loc_select_columns_w_repeats(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[:, ["string_col", "string_col"]].to_pandas()
    pd_result = scalars_pandas_df_index.loc[:, ["string_col", "string_col"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_select_rows_and_columns_w_repeats(
    scalars_df_index, scalars_pandas_df_index
):
    bf_result = scalars_df_index.loc[
        [2, 3, 2], ["string_col", "string_col"]
    ].to_pandas()
    pd_result = scalars_pandas_df_index.loc[[2, 3, 2], ["string_col", "string_col"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_slice_rows_and_select_columns_w_repeats(
    scalars_df_index, scalars_pandas_df_index
):
    bf_result = scalars_df_index.loc[2:5, ["string_col", "string_col"]].to_pandas()
    pd_result = scalars_pandas_df_index.loc[2:5, ["string_col", "string_col"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_bool_series(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[scalars_df_index["bool_col"]].to_pandas()
    pd_result = scalars_pandas_df_index.loc[scalars_pandas_df_index["bool_col"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_list_select_rows_and_columns(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[[2, 3], ["string_col", "int64_col"]].to_pandas()
    pd_result = scalars_pandas_df_index.loc[[2, 3], ["string_col", "int64_col"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_select_column(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[:, "string_col"].to_pandas()
    pd_result = scalars_pandas_df_index.loc[:, "string_col"]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_select_with_column_condition(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[
        scalars_df_index["bool_col"], "string_col"
    ].to_pandas()
    pd_result = scalars_pandas_df_index.loc[
        scalars_pandas_df_index["bool_col"], "string_col"
    ]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_select_with_column_condition_bf_series(
    scalars_df_index, scalars_pandas_df_index
):
    bf_result = scalars_df_index.loc[
        scalars_df_index["bool_col"], scalars_df_index.columns.to_series()
    ].to_pandas()
    pd_result = scalars_pandas_df_index.loc[
        scalars_pandas_df_index["bool_col"],
        scalars_pandas_df_index.columns.to_series(),
    ]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_single_index_with_duplicate(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.set_index("int64_col").loc[2].to_pandas()
    pd_result = scalars_pandas_df_index.set_index("int64_col").loc[2]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_single_index_no_duplicate(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.set_index("int64_col").loc[6].to_pandas()
    pd_result = scalars_pandas_df_index.set_index("int64_col").loc[6]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_setitem_slice_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    bf_df.loc[2:5, "int64_col"] = 99
    pd_df.loc[2:5, "int64_col"] = 99
    assert_pandas_df_equal(bf_df.to_pandas(), pd_df)


def test_loc_setitem_slice_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    bf_series = bf_df["int64_col"] * 2
    pd_series = pd_df["int64_col"] * 2
    bf_df.loc[2:5, "int64_col"] = bf_series
    pd_df.loc[2:5, "int64_col"] = pd_series
    assert_pandas_df_equal(bf_df.to_pandas(), pd_df)


def test_loc_setitem_list_scalar(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    bf_df.loc[[2, 5], "int64_col"] = 99
    pd_df.loc[[2, 5], "int64_col"] = 99
    assert_pandas_df_equal(bf_df.to_pandas(), pd_df)


def test_loc_setitem_list_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    bf_series = bf_df["int64_col"] * 2
    pd_series = pd_df["int64_col"] * 2
    bf_df.loc[[2, 5], "int64_col"] = bf_series
    pd_df.loc[[2, 5], "int64_col"] = pd_series
    assert_pandas_df_equal(bf_df.to_pandas(), pd_df)


@pytest.mark.parametrize(
    ("col", "value"),
    [
        ("new_col", 99),
        ("int64_col", -1),
        ("string_col", "new_string"),
        ("date_col", pd.Timestamp("2024-01-01")),
    ],
)
def test_loc_setitem_bool_series_scalar(scalars_dfs, col, value):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    bf_df.loc[bf_df["bool_col"], col] = value
    pd_df.loc[pd_df["bool_col"], col] = value
    assert_pandas_df_equal(bf_df.to_pandas(), pd_df)


def test_loc_setitem_bool_series_scalar_error(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.copy()
    pd_df = scalars_pandas_df.copy()
    with pytest.raises(TypeError):
        bf_df.loc[bf_df["bool_col"], "int64_col"] = "incompatible_string"
    with pytest.raises(TypeError):
        pd_df.loc[pd_df["bool_col"], "int64_col"] = "incompatible_string"


def test_loc_list_string_index(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.set_index("string_col").loc[["cat", "dog"]].to_pandas()
    pd_result = scalars_pandas_df_index.set_index("string_col").loc[["cat", "dog"]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_list_integer_index(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[[2, 3]].to_pandas()
    pd_result = scalars_pandas_df_index.loc[[2, 3]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_list_multiindex(scalars_dfs):
    # TODO: supply a reason why this isn't compatible with pandas 1.x
    pytest.importorskip("pandas", minversion="2.0.0")
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_df = scalars_df.set_index(["string_col", "int64_col"])
    pd_df = scalars_pandas_df.set_index(["string_col", "int64_col"])
    bf_result = bf_df.loc[[("cat", 2), ("dog", 2)]].to_pandas()
    pd_result = pd_df.loc[[("cat", 2), ("dog", 2)]]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_bf_series_string_index(scalars_df_index, scalars_pandas_df_index):
    bf_result = (
        scalars_df_index.set_index("string_col")
        .loc[scalars_df_index["string_col"]]
        .to_pandas()
    )
    pd_result = scalars_pandas_df_index.set_index("string_col").loc[
        scalars_pandas_df_index["string_col"]
    ]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_bf_series_multiindex(scalars_df_index, scalars_pandas_df_index):
    bf_df = scalars_df_index.set_index(["string_col", "int64_col"])
    pd_df = scalars_pandas_df_index.set_index(["string_col", "int64_col"])
    bf_result = bf_df.loc[bf_df.index.to_series()].to_pandas()
    pd_result = pd_df.loc[pd_df.index.to_series()]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_bf_index_integer_index(scalars_df_index, scalars_pandas_df_index):
    bf_result = scalars_df_index.loc[scalars_df_index.index].to_pandas()
    pd_result = scalars_pandas_df_index.loc[scalars_pandas_df_index.index]
    assert_pandas_df_equal(bf_result, pd_result)


def test_loc_bf_index_integer_index_renamed_col(
    scalars_df_index, scalars_pandas_df_index
):
    bf_df = scalars_df_index.rename(columns={"int64_col": "new_name"})
    pd_df = scalars_pandas_df_index.rename(columns={"int64_col": "new_name"})
    bf_result = bf_df.loc[bf_df.index].to_pandas()
    pd_result = pd_df.loc[pd_df.index]
    assert_pandas_df_equal(bf_result, pd_result)
