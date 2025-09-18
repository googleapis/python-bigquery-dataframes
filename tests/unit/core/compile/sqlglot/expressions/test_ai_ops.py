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

import json

import pytest

from bigframes import dataframe
from bigframes import operations as ops
from bigframes.testing import utils

pytest.importorskip("pytest_snapshot")


def test_ai_generate_bool(scalar_types_df: dataframe.DataFrame, snapshot):
    col_name = "string_col"

    op = ops.AIGenerateBool(
        prompt_context=(None, " is the same as ", None),
        connection_id="test_connection_id",
        endpoint=None,
        request_type="shared",
        model_params=json.dumps(dict()),
    )

    sql = utils._apply_unary_ops(
        scalar_types_df, [op.as_expr(col_name, col_name)], ["result"]
    )

    snapshot.assert_match(sql, "out.sql")
