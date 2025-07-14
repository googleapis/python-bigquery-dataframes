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

import bigframes.bigquery as bbq
import bigframes.pandas as bpd

pytest.importorskip("pytest_snapshot")


def test_isnull(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].isnull()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_notnull(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].notnull()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_invert(scalar_types_df: bpd.DataFrame, snapshot):
    result = ~scalar_types_df["bool_col"]

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_pos(scalar_types_df: bpd.DataFrame, snapshot):
    result = +scalar_types_df["int64_col"]

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_neg(scalar_types_df: bpd.DataFrame, snapshot):
    result = -scalar_types_df["int64_col"]

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_arccos(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].acos()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_arcsin(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].asin()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_cos(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].cos()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_hash(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["string_col"].hash_values()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_sin(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].sin()

    snapshot.assert_match(result.to_frame().sql, "out.sql")


def test_tan(scalar_types_df: bpd.DataFrame, snapshot):
    result = scalar_types_df["int64_col"].tan()

    snapshot.assert_match(result.to_frame().sql, "out.sql")
