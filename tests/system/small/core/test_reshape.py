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
import pandas.testing
import pytest

from bigframes import session
from bigframes.core.reshape import merge


@pytest.mark.parametrize(
    ("left_on", "right_on", "left_index", "right_index"),
    [
        ("col_a", None, False, True),
        (None, "col_c", True, False),
        (None, None, True, True),
    ],
)
def test_join_with_index(
    session: session.Session, left_on, right_on, left_index, right_index
):
    df1 = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [2, 3, 4]})
    bf1 = session.read_pandas(df1)
    df2 = pd.DataFrame({"col_c": [1, 2, 3], "col_d": [2, 3, 4]})
    bf2 = session.read_pandas(df2)

    bf_result = merge.merge(
        bf1,
        bf2,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    ).to_pandas()
    pd_result = pd.merge(
        df1,
        df2,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    )

    pandas.testing.assert_frame_equal(
        bf_result, pd_result, check_dtype=False, check_index_type=False
    )

@pytest.mark.parametrize(
    ("left_on", "right_on", "left_index", "right_index"),
    [
        (["col_a", "col_b"], None, False, True),
        (None, ["col_c", "col_d"], True, False),
        (None, None, True, True),
    ],
)
def test_join_with_multiindex(
    session: session.Session, left_on, right_on, left_index, right_index
):
    multi_idx = pd.MultiIndex.from_tuples([(1,2), (2, 3), (3,4)])
    df1 = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [2, 3, 4]},  index=multi_idx)
    bf1 = session.read_pandas(df1)
    df2 = pd.DataFrame({"col_c": [1, 2, 3], "col_d": [2, 3, 4]}, index=multi_idx)
    bf2 = session.read_pandas(df2)

    bf_result = merge.merge(
        bf1,
        bf2,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    ).to_pandas()
    pd_result = pd.merge(
        df1,
        df2,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
    )

    pandas.testing.assert_frame_equal(
        bf_result, pd_result, check_dtype=False, check_index_type=False
    )
