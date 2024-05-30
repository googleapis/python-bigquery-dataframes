# Copyright 2024 Google LLC
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

import bigframes.exceptions
import bigframes.pandas as bpd
from tests.system.utils import skip_legacy_pandas


def test_null_index_materialize(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index.to_pandas()
    pd.testing.assert_frame_equal(
        bf_result, scalars_pandas_df_default_index, check_index_type=False
    )


def test_null_index_series_repr(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index["int64_too"].head(5).__repr__()
    pd_result = (
        scalars_pandas_df_default_index["int64_too"]
        .head(5)
        .to_string(dtype=True, index=False, length=False, name=True)
    )
    assert bf_result == pd_result


def test_null_index_dataframe_repr(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    bf_result = scalars_df_null_index[["int64_too", "int64_col"]].head(5).__repr__()
    pd_result = (
        scalars_pandas_df_default_index[["int64_too", "int64_col"]]
        .head(5)
        .to_string(index=False)
    )
    assert bf_result == pd_result + "\n\n[5 rows x 2 columns]"


def test_null_index_reset_index(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index.reset_index().to_pandas()
    pd_result = scalars_pandas_df_default_index.reset_index(drop=True)
    pd.testing.assert_frame_equal(bf_result, pd_result, check_index_type=False)


def test_null_index_set_index(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index.set_index("int64_col").to_pandas()
    pd_result = scalars_pandas_df_default_index.set_index("int64_col")
    pd.testing.assert_frame_equal(bf_result, pd_result)


def test_null_index_concat(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = bpd.concat(
        [scalars_df_null_index, scalars_df_null_index], axis=0
    ).to_pandas()
    pd_result = pd.concat(
        [scalars_pandas_df_default_index, scalars_pandas_df_default_index], axis=0
    )
    pd.testing.assert_frame_equal(bf_result, pd_result.reset_index(drop=True))


def test_null_index_aggregate(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index.count().to_pandas()
    pd_result = scalars_pandas_df_default_index.count()

    pd_result.index = pd_result.index.astype("string[pyarrow]")

    pd.testing.assert_series_equal(
        bf_result, pd_result, check_dtype=False, check_index_type=False
    )


def test_null_index_groupby_aggregate(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    bf_result = scalars_df_null_index.groupby("int64_col").count().to_pandas()
    pd_result = scalars_pandas_df_default_index.groupby("int64_col").count()

    pd.testing.assert_frame_equal(bf_result, pd_result, check_dtype=False)


@skip_legacy_pandas
def test_null_index_analytic(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index["int64_col"].cumsum().to_pandas()
    pd_result = scalars_pandas_df_default_index["int64_col"].cumsum()
    pd.testing.assert_series_equal(
        bf_result, pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_groupby_analytic(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    bf_result = (
        scalars_df_null_index.groupby("bool_col")["int64_col"].cummax().to_pandas()
    )
    pd_result = scalars_pandas_df_default_index.groupby("bool_col")[
        "int64_col"
    ].cummax()
    pd.testing.assert_series_equal(
        bf_result, pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_merge_left_null_index_object(
    scalars_df_null_index, scalars_df_default_index, scalars_pandas_df_default_index
):
    df1 = scalars_df_null_index[scalars_df_null_index["int64_col"] > 0]
    df1_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] > 0
    ]
    assert not df1._has_index
    df2 = scalars_df_default_index[scalars_df_default_index["int64_col"] <= 55555]
    df2_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] <= 55555
    ]
    assert df2._has_index

    got = df1.merge(df2, how="inner", on="bool_col")
    expected = df1_pd.merge(df2_pd, how="inner", on="bool_col")

    # Combining any NULL index object should result in a NULL index.
    # This keeps us from generating an index if the user joins a large
    # BigQuery table against small local data, for example.
    assert not got._has_index
    assert got.shape == expected.shape


def test_null_index_merge_right_null_index_object(
    scalars_df_null_index, scalars_df_default_index, scalars_pandas_df_default_index
):
    df1 = scalars_df_default_index[scalars_df_default_index["int64_col"] > 0]
    df1_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] > 0
    ]
    assert df1._has_index
    df2 = scalars_df_null_index[scalars_df_null_index["int64_col"] <= 55555]
    df2_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] <= 55555
    ]
    assert not df2._has_index

    got = df1.merge(df2, how="left", on="bool_col")
    expected = df1_pd.merge(df2_pd, how="left", on="bool_col")

    # Combining any NULL index object should result in a NULL index.
    # This keeps us from generating an index if the user joins a large
    # BigQuery table against small local data, for example.
    assert not got._has_index
    assert got.shape == expected.shape


def test_null_index_merge_two_null_index_objects(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    df1 = scalars_df_null_index[scalars_df_null_index["int64_col"] > 0]
    df1_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] > 0
    ]
    assert not df1._has_index
    df2 = scalars_df_null_index[scalars_df_null_index["int64_col"] <= 55555]
    df2_pd = scalars_pandas_df_default_index[
        scalars_pandas_df_default_index["int64_col"] <= 55555
    ]
    assert not df2._has_index

    got = df1.merge(df2, how="outer", on="bool_col")
    expected = df1_pd.merge(df2_pd, how="outer", on="bool_col")

    assert not got._has_index
    assert got.shape == expected.shape


@skip_legacy_pandas
def test_null_index_stack(scalars_df_null_index, scalars_pandas_df_default_index):
    stacking_cols = ["int64_col", "int64_too"]
    bf_result = scalars_df_null_index[stacking_cols].stack().to_pandas()
    pd_result = (
        scalars_pandas_df_default_index[stacking_cols]
        .stack(future_stack=True)
        .droplevel(level=0, axis=0)
    )
    pd_result.index = pd_result.index.astype(bf_result.index.dtype)
    pd.testing.assert_series_equal(
        bf_result,
        pd_result,
        check_dtype=False,
    )


def test_null_index_series_self_aligns(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    bf_result = scalars_df_null_index["int64_col"] + scalars_df_null_index["int64_too"]
    pd_result = (
        scalars_pandas_df_default_index["int64_col"]
        + scalars_pandas_df_default_index["int64_too"]
    )
    pd.testing.assert_series_equal(
        bf_result.to_pandas(), pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_df_self_aligns(
    scalars_df_null_index, scalars_pandas_df_default_index
):
    bf_result = (
        scalars_df_null_index[["int64_col", "float64_col"]]
        + scalars_df_null_index[["int64_col", "float64_col"]]
    )
    pd_result = (
        scalars_pandas_df_default_index[["int64_col", "float64_col"]]
        + scalars_pandas_df_default_index[["int64_col", "float64_col"]]
    )
    pd.testing.assert_frame_equal(
        bf_result.to_pandas(), pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_setitem(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = scalars_df_null_index.copy()
    bf_result["new_col"] = (
        scalars_df_null_index["int64_col"] + scalars_df_null_index["float64_col"]
    )
    pd_result = scalars_pandas_df_default_index.copy()
    pd_result["new_col"] = (
        scalars_pandas_df_default_index["int64_col"]
        + scalars_pandas_df_default_index["float64_col"]
    )
    pd.testing.assert_frame_equal(
        bf_result.to_pandas(), pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_df_concat(scalars_df_null_index, scalars_pandas_df_default_index):
    bf_result = bpd.concat([scalars_df_null_index, scalars_df_null_index])
    pd_result = pd.concat(
        [scalars_pandas_df_default_index, scalars_pandas_df_default_index]
    )
    pd.testing.assert_frame_equal(
        bf_result.to_pandas(), pd_result.reset_index(drop=True), check_dtype=False
    )


def test_null_index_align_error(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        _ = (
            scalars_df_null_index["int64_col"]
            + scalars_df_null_index["int64_col"].cumsum()
        )


def test_null_index_loc_error(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        scalars_df_null_index["int64_col"].loc[1]


def test_null_index_at_error(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        scalars_df_null_index["int64_col"].at[1]


def test_null_index_idxmin_error(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        scalars_df_null_index[["int64_col", "int64_too"]].idxmin()


def test_null_index_index_property(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        _ = scalars_df_null_index.index


def test_null_index_transpose(scalars_df_null_index):
    with pytest.raises(bigframes.exceptions.NullIndexError):
        _ = scalars_df_null_index.T
