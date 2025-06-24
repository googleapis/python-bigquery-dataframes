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

import pytest

import bigframes.pandas as bpd

pytest.importorskip("pytest_snapshot")


@pytest.mark.parametrize(
    ("how"),
    ["left", "right", "outer", "inner"],
)
def test_compile_join(scalars_types_df: bpd.DataFrame, how, snapshot):
    left = scalars_types_df[["int64_col"]]
    right = scalars_types_df.set_index("int64_col")[["int64_too"]]
    join = left.join(right, how=how)
    snapshot.assert_match(join.sql, "out.sql")


def test_compile_join_w_on(scalars_types_df: bpd.DataFrame, snapshot):
    selected_cols = ["int64_col", "int64_too"]
    left = scalars_types_df[selected_cols]
    right = (
        scalars_types_df[selected_cols]
        .rename(columns={"int64_col": "col1", "int64_too": "col2"})
        .set_index("col2")
    )
    join = left.join(right, on="int64_too")
    snapshot.assert_match(join.sql, "out.sql")


def test_compile_join_by_cross(compiler_session, snapshot):
    df1 = bpd.DataFrame(
        {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]},
        session=compiler_session,
    )
    df2 = bpd.DataFrame(
        {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]},
        session=compiler_session,
    )
    merge = df1.merge(df2, left_on="lkey", right_on="rkey", how="cross")
    snapshot.assert_match(merge.sql, "out.sql")
