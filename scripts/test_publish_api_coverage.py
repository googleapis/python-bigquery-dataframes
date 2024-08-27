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

import pandas

from . import publish_api_coverage


def test_api_coverage_produces_expected_schema():
    df = publish_api_coverage.build_api_coverage_table("my_bf_ver", "my_release_ver")
    pandas.testing.assert_series_equal(
        df.dtypes,
        pandas.Series(
            data={
                # Note to developer: if you update this test, you will also
                # need to update schema of the API coverage BigQuery table in
                # the bigframes-metrics project.
                "api": "string",
                "pattern": "string",
                "kind": "string",
                "is_in_bigframes": "boolean",
                "missing_parameters": "string",
                "requires_index": "string",
                "requires_ordering": "string",
                "module": "string",
                "timestamp": "datetime64[us]",
                "bigframes_version": "string",
                "release_version": "string",
            },
        ),
    )
