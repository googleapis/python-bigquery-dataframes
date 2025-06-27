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

from __future__ import annotations

import pandas
import pyarrow
import pytest

from bigframes.pandas.io import api as module_under_test


@pytest.mark.parametrize(
    ("df_pd",),
    (
        # Regression tests for b/428190014.
        #
        # Test every BigQuery type we support, especially those where the legacy
        # SQL type name differs from the GoogleSQL type name.
        #
        # See:
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types
        # and compare to the legacy types at
        # https://cloud.google.com/bigquery/docs/data-types
        pytest.param(
            pandas.DataFrame(
                {
                    "ints": pandas.Series(
                        [[1], [2], [3]],
                        dtype=pandas.ArrowDtype(pyarrow.list_(pyarrow.int64())),
                    ),
                    "floats": pandas.Series(
                        [[1.0], [2.0], [3.0]],
                        dtype=pandas.ArrowDtype(pyarrow.list_(pyarrow.float64())),
                    ),
                }
            ),
            id="arrays",
        ),
        pytest.param(
            pandas.DataFrame(
                {
                    "bool": pandas.Series([True, False, True], dtype="bool"),
                    "boolean": pandas.Series([True, None, True], dtype="boolean"),
                    "object": pandas.Series([True, None, True], dtype="object"),
                    "arrow_bool": pandas.Series(
                        [True, None, True], dtype=pandas.ArrowDtype(pyarrow.bool_())
                    ),
                }
            ),
            id="bools",
        ),
    ),
)
def test_read_gbq_colab_sessionless_dry_run_generates_valid_sql_for_local_dataframe(
    df_pd: pandas.DataFrame,
):
    # This method will fail with an exception if it receives invalid SQL.
    result = module_under_test._run_read_gbq_colab_sessionless_dry_run(
        query="SELECT * FROM {df_pd}",
        pyformat_args={"df_pd": df_pd},
    )
    assert isinstance(result, pandas.Series)
