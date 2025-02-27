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


import warnings

import google.api_core.exceptions
import pytest

import bigframes

WIKIPEDIA_TABLE = "bigquery-public-data.samples.wikipedia"
LARGE_TABLE_OPTION = "bigquery.allow_large_results"


def test_to_pandas_batches_raise_when_large_result_not_allowed(session):
    with bigframes.option_context(LARGE_TABLE_OPTION, False), pytest.raises(
        google.api_core.exceptions.Forbidden
    ):
        df = session.read_gbq(WIKIPEDIA_TABLE)
        next(df.to_pandas_batches(page_size=500, max_results=1500))


def test_to_pandas_batches_override_global_option(
    session,
):
    with bigframes.option_context(LARGE_TABLE_OPTION, False):
        df = session.read_gbq(WIKIPEDIA_TABLE)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            next(
                df.to_pandas_batches(
                    page_size=500, max_results=1500, allow_large_results=True
                )
            )
            assert len(w) == 2
            assert issubclass(w[0].category, FutureWarning)
            assert str(w[0].message).startswith(
                "The query result size has exceeded 10 GB."
            )


def test_to_pandas_raise_when_large_result_not_allowed(session):
    with bigframes.option_context(LARGE_TABLE_OPTION, False), pytest.raises(
        google.api_core.exceptions.Forbidden
    ):
        df = session.read_gbq(WIKIPEDIA_TABLE)
        next(df.to_pandas())
