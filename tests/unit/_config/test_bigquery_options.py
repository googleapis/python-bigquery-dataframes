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

import re

import pytest

import bigframes._config.bigquery_options as bigquery_options


@pytest.mark.parametrize(
    [
        "attribute",
    ],
    [
        (attribute,)
        for attribute in [
            "credentials",
            "location",
            "project",
            "remote_udf_connection",
        ]
    ],
)
def test_setter_raises_if_session_started(attribute):
    options = bigquery_options.BigQueryOptions()
    original_object = object()
    setattr(options, attribute, original_object)
    second_object = object()
    assert getattr(options, attribute) is original_object
    assert getattr(options, attribute) is not second_object

    options._session_started = True
    expected_message = re.escape(
        bigquery_options.SESSION_STARTED_MESSAGE.format(attribute=attribute)
    )
    with pytest.raises(ValueError, match=expected_message):
        setattr(options, attribute, original_object)

    assert getattr(options, attribute) is original_object
    assert getattr(options, attribute) is not second_object
