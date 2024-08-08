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

import itertools

import pandas as pd
from pandas.testing import assert_frame_equal

import bigframes.core.indexes

index_to_frame_test_args = [[True, False], [None, "food"]]
multi_index_to_frame_test_args = [[True, False], [None, ["x", "y"]]]


def test_index_repr_with_uninitialized_object():
    """Ensures Index.__init__ can be paused in a visual debugger without crashing.

    Regression test for https://github.com/googleapis/python-bigquery-dataframes/issues/728
    """
    # Avoid calling __init__ to simulate pausing __init__ in a debugger.
    # https://stackoverflow.com/a/6384982/101923
    index = object.__new__(bigframes.core.indexes.Index)
    got = repr(index)
    assert "Index" in got


def test_multiindex_repr_with_uninitialized_object():
    """Ensures MultiIndex.__init__ can be paused in a visual debugger without crashing.

    Regression test for https://github.com/googleapis/python-bigquery-dataframes/issues/728
    """
    # Avoid calling __init__ to simulate pausing __init__ in a debugger.
    # https://stackoverflow.com/a/6384982/101923
    index = object.__new__(bigframes.core.indexes.MultiIndex)
    got = repr(index)
    assert "MultiIndex" in got


def test_index_to_frame():
    pd_idx = pd.Index(["Ant", "Bear", "Cow"], name="animal", dtype="string[pyarrow]")
    bf_idx = bigframes.core.indexes.Index(["Ant", "Bear", "Cow"], name="animal")

    for index_arg, name_arg in itertools.product(*index_to_frame_test_args):
        if name_arg is None:
            pd_df = pd_idx.to_frame(index=index_arg)
            bf_df = bf_idx.to_frame(index=index_arg)
        else:
            pd_df = pd_idx.to_frame(index=index_arg, name=name_arg)
            bf_df = bf_idx.to_frame(index=index_arg, name=name_arg)
        print(pd_df)
        print(bf_df.to_pandas())
        assert_frame_equal(
            pd_df, bf_df.to_pandas(), check_column_type=False, check_index_type=False
        )
        # BigFrames type casting is weird
        # automatically casts dtype to string whereas pandas dtype is object
        # additionally, pandas uses string[python] and BigFrames uses string[pyarrow]
        # so we set dtype in pandas index creation
        # similarly, pandas uses int64 dtype for numerical index and BigFrames uses Int64


def test_multi_index_to_frame():
    pd_idx = pd.MultiIndex.from_arrays([["a", "b", "c"], ["d", "e", "f"]])
    bf_idx = bigframes.core.indexes.MultiIndex.from_arrays(
        [["a", "b", "c"], ["d", "e", "f"]]
    )
    for index_arg, name_arg in itertools.product(*multi_index_to_frame_test_args):
        if name_arg is None:
            pd_df = pd_idx.to_frame(index=index_arg)
            bf_df = bf_idx.to_frame(index=index_arg)
        else:
            pd_df = pd_idx.to_frame(index=index_arg, name=name_arg)
            bf_df = bf_idx.to_frame(index=index_arg, name=name_arg)
        print(pd_df)
        print(bf_df.to_pandas())
        assert_frame_equal(
            pd_df,
            bf_df.to_pandas(),
            check_dtype=False,
            check_column_type=False,
            check_index_type=False,
        )
