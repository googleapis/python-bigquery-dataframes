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

import bigframes
import bigframes.pandas as bpd

pytest.importorskip("pytest_snapshot")


def test_compile_readlocal(
    scalars_types_pandas_df: pd.DataFrame, compiler_session: bigframes.Session, snapshot
):
    bf_df = bpd.DataFrame(scalars_types_pandas_df, session=compiler_session)
    snapshot.assert_match(bf_df.sql, "out.sql")


def test_compile_readlocal_w_structs_df(
    nested_structs_pandas_df: pd.DataFrame,
    compiler_session: bigframes.Session,
    snapshot,
):
    bf_df = bpd.DataFrame(nested_structs_pandas_df, session=compiler_session)
    snapshot.assert_match(bf_df.sql, "out.sql")


def test_compile_readlocal_w_lists_df(
    repeated_pandas_df: pd.DataFrame,
    compiler_session: bigframes.Session,
    snapshot,
):
    bf_df = bpd.DataFrame(repeated_pandas_df, session=compiler_session)
    snapshot.assert_match(bf_df.sql, "out.sql")


def test_compile_readlocal_w_json_df(
    json_pandas_df: pd.DataFrame,
    compiler_session: bigframes.Session,
    snapshot,
):
    bf_df = bpd.DataFrame(json_pandas_df, session=compiler_session)
    snapshot.assert_match(bf_df.sql, "out.sql")
