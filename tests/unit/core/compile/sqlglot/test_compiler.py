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

import google.cloud.bigquery as bigquery
import pandas as pd
import pytest

import bigframes.core
import bigframes.core.nodes as nodes
import bigframes.pandas as bpd


def test_compile_local(
    inline_pd_df: pd.DataFrame, sql_compiler_session: bigframes.Session
):
    bf_df = bpd.DataFrame(inline_pd_df, session=sql_compiler_session)
    expected_sql = "SELECT `column_0` AS `int1`, `column_1` AS `int2`, `column_2` AS `bools`, `column_3` AS `strings`, `bfuid_col_1` AS `bfuid_col_2` FROM UNNEST(ARRAY<STRUCT<`column_0` INT64, `column_1` INT64, `column_2` BOOLEAN, `column_3` STRING, `bfuid_col_1` INT64>>[(1, -10, TRUE, 'b', 0), (2, 20, CAST(NULL AS BOOLEAN), 'aa', 1), (3, 30, FALSE, 'ccc', 2)]) ORDER BY `bfuid_col_2` ASC NULLS LAST"
    assert bf_df.sql == expected_sql


def test_compile_add(
    inline_pd_df: pd.DataFrame, sql_compiler_session: bigframes.Session
):
    bf_df = bpd.DataFrame(inline_pd_df, session=sql_compiler_session)
    bf_add = bf_df["int1"] + bf_df["int2"]
    expected_sql = "SELECT `bfuid_col_4` AS `bigframes_unnamed_column`, `bfuid_col_6` AS `bfuid_col_7` FROM UNNEST(ARRAY<STRUCT<`column_0` INT64, `column_1` INT64, `bfuid_col_5` INT64>>[(1, -10, 0), (2, 20, 1), (3, 30, 2)]) ORDER BY `bfuid_col_7` ASC NULLS LAST"
    assert bf_add.sql == expected_sql


def test_compile_order_by(
    inline_pd_df: pd.DataFrame, sql_compiler_session: bigframes.Session
):
    bf_df = bpd.DataFrame(inline_pd_df, session=sql_compiler_session)
    bf_sort_values = bf_df.sort_values("strings")
    expected_sql = "SELECT `column_0` AS `int1`, `column_1` AS `int2`, `column_2` AS `bools`, `column_3` AS `strings`, `bfuid_col_1` AS `bfuid_col_2` FROM UNNEST(ARRAY<STRUCT<`column_0` INT64, `column_1` INT64, `column_2` BOOLEAN, `column_3` STRING, `bfuid_col_1` INT64>>[(1, -10, TRUE, 'b', 0), (2, 20, CAST(NULL AS BOOLEAN), 'aa', 1), (3, 30, FALSE, 'ccc', 2)]) ORDER BY `strings` ASC NULLS LAST, `bfuid_col_2` ASC NULLS LAST"
    assert bf_sort_values.sql == expected_sql


def test_compile_filter(
    inline_pd_df: pd.DataFrame, sql_compiler_session: bigframes.Session
):
    bf_df = bpd.DataFrame(inline_pd_df, session=sql_compiler_session)
    bf_filter = bf_df[bf_df["int2"] >= 1]
    expected_sql = "SELECT `column_0` AS `int1`, `column_1` AS `int2`, `column_2` AS `bools`, `column_3` AS `strings`, `bfuid_col_1` AS `bfuid_col_2` FROM UNNEST(ARRAY<STRUCT<`column_0` INT64, `column_1` INT64, `column_2` BOOLEAN, `column_3` STRING, `bfuid_col_1` INT64>>[(1, -10, TRUE, 'b', 0), (2, 20, CAST(NULL AS BOOLEAN), 'aa', 1), (3, 30, FALSE, 'ccc', 2)]) ORDER BY `strings` ASC NULLS LAST, `bfuid_col_2` ASC NULLS LAST"
    assert bf_filter.sql == expected_sql
