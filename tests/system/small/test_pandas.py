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

import pandas as pd
import pytest

import bigframes.pandas as bpd
from tests.system.utils import assert_pandas_df_equal_ignore_ordering


def test_concat_dataframe(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = bpd.concat(11 * [scalars_df])
    bf_result = bf_result.to_pandas()
    pd_result = pd.concat(11 * [scalars_pandas_df])

    pd.testing.assert_frame_equal(bf_result, pd_result)


def test_concat_series(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = bpd.concat(
        [scalars_df.int64_col, scalars_df.int64_too, scalars_df.int64_col]
    )
    bf_result = bf_result.to_pandas()
    pd_result = pd.concat(
        [
            scalars_pandas_df.int64_col,
            scalars_pandas_df.int64_too,
            scalars_pandas_df.int64_col,
        ]
    )

    pd.testing.assert_series_equal(bf_result, pd_result)


@pytest.mark.parametrize(
    ("how"),
    [
        ("inner"),
        ("outer"),
    ],
)
def test_concat_dataframe_mismatched_columns(scalars_dfs, how):
    cols1 = ["int64_too", "int64_col", "float64_col"]
    cols2 = ["int64_col", "string_col", "int64_too"]
    scalars_df, scalars_pandas_df = scalars_dfs
    bf_result = bpd.concat([scalars_df[cols1], scalars_df[cols2]], join=how)
    bf_result = bf_result.to_pandas()
    pd_result = pd.concat(
        [scalars_pandas_df[cols1], scalars_pandas_df[cols2]],
        join=how,
    )

    pd.testing.assert_frame_equal(bf_result, pd_result)


@pytest.mark.parametrize(
    ("how",),
    [
        ("inner",),
        ("outer",),
    ],
)
def test_concat_axis_1(scalars_dfs, how):
    if pd.__version__.startswith("1."):
        pytest.skip("pandas has different behavior in 1.x")
    scalars_df, scalars_pandas_df = scalars_dfs
    cols1 = ["int64_col", "float64_col", "rowindex_2"]
    cols2 = ["int64_col", "bool_col", "string_col", "rowindex_2"]

    part1 = scalars_df[cols1]
    part1.index.name = "newindexname"
    # Offset the rows somewhat so that outer join can have an effect.
    part2 = (
        scalars_df[cols2]
        .assign(rowindex_2=scalars_df["rowindex_2"] + 2)
        .sort_values(["string_col"], kind="stable")
    )
    part3 = scalars_df["int64_too"].cumsum().iloc[2:]

    bf_result = bpd.concat([part1, part2, part3], join=how, axis=1)

    # Copy since modifying index
    pd_part1 = scalars_pandas_df.copy()[cols1]
    pd_part1.index.name = "newindexname"
    # Offset the rows somewhat so that outer join can have an effect.
    pd_part2 = (
        scalars_pandas_df[cols2]
        .assign(rowindex_2=scalars_pandas_df["rowindex_2"] + 2)
        .sort_values(["string_col"], kind="stable")
    )
    pd_part3 = scalars_pandas_df["int64_too"].cumsum().iloc[2:]

    pd_result = pd.concat([pd_part1, pd_part2, pd_part3], join=how, axis=1)

    pd.testing.assert_frame_equal(bf_result.to_pandas(), pd_result)


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
    on = "rowindex_2"
    left_columns = ["int64_col", "float64_col", "rowindex_2"]
    right_columns = ["int64_col", "bool_col", "string_col", "rowindex_2"]

    left = scalars_df[left_columns]
    # Offset the rows somewhat so that outer join can have an effect.
    right = scalars_df[right_columns].assign(rowindex_2=scalars_df["rowindex_2"] + 2)

    df = bpd.merge(left, right, merge_how, on, sort=True)
    bf_result = df.to_pandas()

    pd_result = pd.merge(
        scalars_pandas_df[left_columns],
        scalars_pandas_df[right_columns].assign(
            rowindex_2=scalars_pandas_df["rowindex_2"] + 2
        ),
        merge_how,
        on,
        sort=True,
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
def test_merge_left_on_right_on(scalars_dfs, merge_how):
    scalars_df, scalars_pandas_df = scalars_dfs
    left_columns = ["int64_col", "float64_col", "int64_too"]
    right_columns = ["int64_col", "bool_col", "string_col", "rowindex_2"]

    left = scalars_df[left_columns]
    right = scalars_df[right_columns]

    df = bpd.merge(
        left, right, merge_how, left_on="int64_too", right_on="rowindex_2", sort=True
    )
    bf_result = df.to_pandas()

    pd_result = pd.merge(
        scalars_pandas_df[left_columns],
        scalars_pandas_df[right_columns],
        merge_how,
        left_on="int64_too",
        right_on="rowindex_2",
        sort=True,
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
def test_merge_series(scalars_dfs, merge_how):
    scalars_df, scalars_pandas_df = scalars_dfs
    left_column = "int64_too"
    right_columns = ["int64_col", "bool_col", "string_col", "rowindex_2"]

    left = scalars_df[left_column]
    right = scalars_df[right_columns]

    df = bpd.merge(
        left, right, merge_how, left_on="int64_too", right_on="rowindex_2", sort=True
    )
    bf_result = df.to_pandas()

    pd_result = pd.merge(
        scalars_pandas_df[left_column],
        scalars_pandas_df[right_columns],
        merge_how,
        left_on="int64_too",
        right_on="rowindex_2",
        sort=True,
    )

    assert_pandas_df_equal_ignore_ordering(bf_result, pd_result)


def test_cut(scalars_dfs):
    scalars_df, scalars_pandas_df = scalars_dfs

    pd_result = pd.cut(scalars_pandas_df["float64_col"], 5, labels=False)
    bf_result = bpd.cut(scalars_df["float64_col"], 5, labels=False)

    # make sure the result is a supported dtype
    assert bf_result.dtype == bpd.Int64Dtype()

    bf_result = bf_result.to_pandas()
    pd_result = pd_result.astype("Int64")
    pd.testing.assert_series_equal(bf_result, pd_result)
