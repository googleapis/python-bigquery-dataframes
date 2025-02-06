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

import pandas
import pytest

import bigframes.pandas as bpd
from bigframes.functions import _function_session as bff_session
from tests.system.large.functions import function_utils

bpd.options.experiments.udf = True


@pytest.fixture(scope="module")
def bq_cf_connection() -> str:
    """Pre-created BQ connection in the test project in US location, used to
    invoke cloud function.

    $ bq show --connection --location=us --project_id=PROJECT_ID bigframes-rf-conn
    """
    return "bigframes-rf-conn"


@pytest.mark.flaky(retries=2, delay=120)
@pytest.mark.skipif(
    sys.version_info[:2] in bff_session._MANAGED_FUNC_PYTHON_VERSIONS,
    reason=f"Supported version: {bff_session._MANAGED_FUNC_PYTHON_VERSIONS}",
)
def test_managed_function_multiply_with_ibis(
    session,
    scalars_table_id,
    bigquery_client,
    ibis_client,
    dataset_id,
    bq_cf_connection,
):

    try:

        @session.udf(
            [int, int],
            int,
            dataset_id,
            bq_cf_connection,
        )
        def multiply(x, y):
            return x * y

        _, dataset_name, table_name = scalars_table_id.split(".")
        if not ibis_client.dataset:
            ibis_client.dataset = dataset_name

        col_name = "int64_col"
        table = ibis_client.tables[table_name]
        table = table.filter(table[col_name].notnull()).order_by("rowindex").head(10)
        sql = table.compile()
        pandas_df_orig = bigquery_client.query(sql).to_dataframe()

        col = table[col_name]
        col_2x = multiply(col, 2).name("int64_col_2x")
        col_square = multiply(col, col).name("int64_col_square")
        table = table.mutate([col_2x, col_square])
        sql = table.compile()
        pandas_df_new = bigquery_client.query(sql).to_dataframe()

        pandas.testing.assert_series_equal(
            pandas_df_orig[col_name] * 2,
            pandas_df_new["int64_col_2x"],
            check_names=False,
        )

        pandas.testing.assert_series_equal(
            pandas_df_orig[col_name] * pandas_df_orig[col_name],
            pandas_df_new["int64_col_square"],
            check_names=False,
        )
    finally:
        # clean up the gcp assets created for the managed function.
        function_utils.cleanup_function_assets(multiply, bigquery_client)
