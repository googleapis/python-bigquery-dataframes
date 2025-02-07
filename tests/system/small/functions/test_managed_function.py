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

import sys

import pandas as pd
import pytest

from bigframes.functions import _function_session as bff_session
from bigframes.functions import function as bff
import bigframes.pandas as bpd
from tests.system.small.functions import function_utils
from tests.system.utils import assert_pandas_df_equal

bpd.options.experiments.udf = True


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection in the test project in US location, used to
    invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "bigframes-rf-conn"


@pytest.mark.skipif(
    sys.version_info[:2] not in bff_session._MANAGED_FUNC_PYTHON_VERSIONS,
    reason=f"Supported version: {bff_session._MANAGED_FUNC_PYTHON_VERSIONS}",
)
@pytest.mark.flaky(retries=2, delay=120)
def test_managed_function_direct_no_session_param(
    bigquery_client,
    bigqueryconnection_client,
    resourcemanager_client,
    scalars_dfs,
    dataset_id_permanent,
    bq_cf_connection,
):
    def square(x):
        return x * x

    square = bff.udf(
        int,
        int,
        bigquery_client=bigquery_client,
        bigquery_connection_client=bigqueryconnection_client,
        resource_manager_client=resourcemanager_client,
        dataset=dataset_id_permanent,
        bigquery_connection=bq_cf_connection,
        name=function_utils.get_function_name(square),
    )(square)

    # Function should still work normally.
    assert square(2) == 4

    assert hasattr(square, "bigframes_function")
    assert hasattr(square, "ibis_node")

    scalars_df, scalars_pandas_df = scalars_dfs

    bf_int64_col = scalars_df["int64_col"]
    bf_int64_col_filter = bf_int64_col.notnull()
    bf_int64_col_filtered = bf_int64_col[bf_int64_col_filter]
    bf_result_col = bf_int64_col_filtered.apply(square)
    bf_result = (
        bf_int64_col_filtered.to_frame().assign(result=bf_result_col).to_pandas()
    )

    pd_int64_col = scalars_pandas_df["int64_col"]
    pd_int64_col_filter = pd_int64_col.notnull()
    pd_int64_col_filtered = pd_int64_col[pd_int64_col_filter]
    pd_result_col = pd_int64_col_filtered.apply(lambda x: x * x)
    pd_result_col = pd_result_col.astype(pd.Int64Dtype())
    pd_result = pd_int64_col_filtered.to_frame().assign(result=pd_result_col)

    assert_pandas_df_equal(bf_result, pd_result)
