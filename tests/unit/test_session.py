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

import google.api_core.exceptions
import pytest


@pytest.mark.parametrize("missing_parts_table_id", [(""), ("table")])
def test_read_gbq_missing_parts(session, missing_parts_table_id):
    with pytest.raises(ValueError):
        session.read_gbq(missing_parts_table_id)


@pytest.mark.parametrize(
    "not_found_table_id",
    [("unknown.dataset.table"), ("project.unknown.table"), ("project.dataset.unknown")],
)
def test_read_gdb_not_found_tables(session, not_found_table_id):
    with pytest.raises(google.api_core.exceptions.NotFound):
        session.read_gbq(not_found_table_id)


@pytest.mark.parametrize(
    "good_table_id,expected", [("project.dataset.table", 1), ("dataset.table", 0)]
)
def test_read_gbq_good_tables(session, scalars_pandas_df, good_table_id, expected):
    index_cols = ("rowindex",) if scalars_pandas_df.index.name == "rowindex" else ()
    df = session.read_gbq(good_table_id, index_cols=index_cols)
    assert len(df.columns) == expected


def test_read_gbq_w_col_order(session, scalars_pandas_df):
    scalars_table_id = "project.dataset.scalars_table"
    index_cols = ("rowindex",) if scalars_pandas_df.index.name == "rowindex" else ()
    df = session.read_gbq(scalars_table_id, index_cols=index_cols)
    assert len(df.columns) == 4

    df = session.read_gbq(
        scalars_table_id, col_order=["bool_col"], index_cols=index_cols
    )
    assert len(df.columns) == 1

    with pytest.raises(ValueError):
        df = session.read_gbq(
            scalars_table_id, col_order=["unknown"], index_cols=index_cols
        )
