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

import bigframes
import bigframes.series


def test_to_pandas_override_global_option(scalars_df_index):
    with bigframes.option_context("bigquery.allow_large_results", True):

        bf_series = scalars_df_index["int64_col"]

        # Direct call to_pandas uses global default setting (allow_large_results=True)
        bf_series.to_pandas()
        table_id = bf_series._query_job.destination.table_id
        assert table_id is not None

        session = bf_series._block.session
        execution_count = session._metrics.execution_count

        # When allow_large_results=False, a query_job object should not be created.
        # Therefore, the table_id should remain unchanged.
        bf_series.to_pandas(allow_large_results=False)
        assert bf_series._query_job.destination.table_id == table_id
        assert session._metrics.execution_count - execution_count == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"sampling_method": "head"},
            r"DEPRECATED[\S\s]*sampling_method[\S\s]*Series",
            id="sampling_method",
        ),
        pytest.param(
            {"random_state": 10},
            r"DEPRECATED[\S\s]*random_state[\S\s]*Series",
            id="random_state",
        ),
        pytest.param(
            {"max_download_size": 10},
            r"DEPRECATED[\S\s]*max_download_size[\S\s]*Series",
            id="max_download_size",
        ),
    ],
)
def test_to_pandas_warns_deprecated_parameters(scalars_df_index, kwargs, message):
    s: bigframes.series.Series = scalars_df_index["int64_col"]
    with pytest.warns(UserWarning, match=message):
        s.to_pandas(
            # limits only apply for allow_large_result=True
            allow_large_results=True,
            **kwargs,
        )
