# Copyright 2024 Google LLC
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

from unittest import mock

from google.cloud import bigquery
import pytest
import pytest_mock

import bigframes
from bigframes.ml import core

TEMP_MODEL_ID = bigquery.ModelReference.from_string(
    "test-project._anon123.temp_model_id"
)


@pytest.fixture
def mock_session():
    mock_session = mock.create_autospec(spec=bigframes.Session)

    mock_session._anonymous_dataset = bigquery.DatasetReference(
        TEMP_MODEL_ID.project, TEMP_MODEL_ID.dataset_id
    )
    mock_session._bq_kms_key_name = None
    mock_session._metrics = None

    query_job = mock.create_autospec(bigquery.QueryJob)
    type(query_job).destination = mock.PropertyMock(
        return_value=bigquery.TableReference(
            mock_session._anonymous_dataset, TEMP_MODEL_ID.model_id
        )
    )
    mock_session._start_query_ml_ddl.return_value = (None, query_job)

    return mock_session


@pytest.fixture
def bqml_model_factory(mocker: pytest_mock.MockerFixture):
    mocker.patch(
        "bigframes.ml.core.BqmlModelFactory._create_model_ref",
        return_value=TEMP_MODEL_ID,
    )
    bqml_model_factory = core.BqmlModelFactory()

    return bqml_model_factory
