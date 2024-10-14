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

import dataclasses
import datetime
from typing import Mapping, Optional, Sequence, Union
import unittest.mock as mock
import weakref

import google.auth.credentials
import google.cloud.bigquery
import pytest

import bigframes
import bigframes.clients
import bigframes.core.compile.polars
import bigframes.core.ordering
import bigframes.dataframe
import bigframes.session.clients
import bigframes.session.executor
import bigframes.session.metrics

"""Utilities for creating test resources."""


TEST_SCHEMA = (google.cloud.bigquery.SchemaField("col", "INTEGER"),)


@dataclasses.dataclass
class TestExecutor(bigframes.session.executor.Executor):
    compiler = bigframes.core.compile.polars.PolarsCompiler()

    def execute(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        ordered: bool = True,
        col_id_overrides: Mapping[str, str] = {},
        use_explicit_destination: bool = False,
        get_size_bytes: bool = False,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
    ):
        """
        Execute the ArrayValue, storing the result to a temporary session-owned table.
        """
        import polars

        lazy_frame: polars.LazyFrame = self.compiler.compile(array_value)
        pa_table = lazy_frame.collect().to_arrow()
        # Currently, pyarrow types might not quite be exactly the ones in the bigframes schema.
        # Nullability may be different, and might use large versions of list, string datatypes.
        return bigframes.session.executor.ExecuteResult(
            arrow_batches=lambda: pa_table.to_batches(),
            schema=array_value.schema,
            total_bytes=pa_table.nbytes,
            total_rows=pa_table.num_rows,
        )


class TestSession(bigframes.session.Session):
    def __init__(self):
        self._location = None  # type: ignore
        self._bq_kms_key_name = None  # type: ignore
        self._clients_provider = None  # type: ignore
        self.ibis_client = None  # type: ignore
        self._bq_connection = None  # type: ignore
        self._skip_bq_connection_check = True
        self._session_id: str = "test_session"
        self._objects: list[
            weakref.ReferenceType[
                Union[
                    bigframes.core.indexes.Index,
                    bigframes.series.Series,
                    bigframes.dataframe.DataFrame,
                ]
            ]
        ] = []
        self._strictly_ordered: bool = True
        self._allow_ambiguity = False  # type: ignore
        self._default_index_type = bigframes.enums.DefaultIndexKind.SEQUENTIAL_INT64
        self._metrics = bigframes.session.metrics.ExecutionMetrics()
        self._remote_function_session = None  # type: ignore
        self._temp_storage_manager = None  # type: ignore
        self._executor = TestExecutor()
        self._loader = None  # type: ignore


def create_bigquery_session(
    bqclient: Optional[mock.Mock] = None,
    session_id: str = "abcxyz",
    table_schema: Sequence[google.cloud.bigquery.SchemaField] = TEST_SCHEMA,
    anonymous_dataset: Optional[google.cloud.bigquery.DatasetReference] = None,
    location: str = "test-region",
) -> bigframes.Session:
    credentials = mock.create_autospec(
        google.auth.credentials.Credentials, instance=True
    )

    if anonymous_dataset is None:
        anonymous_dataset = google.cloud.bigquery.DatasetReference(
            "test-project",
            "test_dataset",
        )

    if bqclient is None:
        bqclient = mock.create_autospec(google.cloud.bigquery.Client, instance=True)
        bqclient.project = "test-project"
        bqclient.location = location

        # Mock the location.
        table = mock.create_autospec(google.cloud.bigquery.Table, instance=True)
        table._properties = {}
        type(table).location = mock.PropertyMock(return_value=location)
        type(table).schema = mock.PropertyMock(return_value=table_schema)
        type(table).reference = mock.PropertyMock(
            return_value=anonymous_dataset.table("test_table")
        )
        type(table).num_rows = mock.PropertyMock(return_value=1000000000)
        bqclient.get_table.return_value = table

    if anonymous_dataset is None:
        anonymous_dataset = google.cloud.bigquery.DatasetReference(
            "test-project",
            "test_dataset",
        )

    def query_mock(query, *args, **kwargs):
        query_job = mock.create_autospec(google.cloud.bigquery.QueryJob)
        type(query_job).destination = mock.PropertyMock(
            return_value=anonymous_dataset.table("test_table"),
        )
        type(query_job).session_info = google.cloud.bigquery.SessionInfo(
            {"sessionInfo": {"sessionId": session_id}},
        )

        if query.startswith("SELECT CURRENT_TIMESTAMP()"):
            query_job.result = mock.MagicMock(return_value=[[datetime.datetime.now()]])
        else:
            type(query_job).schema = mock.PropertyMock(return_value=table_schema)

        return query_job

    bqclient.query = query_mock

    clients_provider = mock.create_autospec(bigframes.session.clients.ClientsProvider)
    type(clients_provider).bqclient = mock.PropertyMock(return_value=bqclient)
    clients_provider._credentials = credentials

    bqoptions = bigframes.BigQueryOptions(credentials=credentials, location=location)
    session = bigframes.Session(context=bqoptions, clients_provider=clients_provider)
    session._bq_connection_manager = mock.create_autospec(
        bigframes.clients.BqConnectionManager, instance=True
    )
    return session


def create_dataframe(
    monkeypatch: pytest.MonkeyPatch, session: Optional[bigframes.Session] = None
) -> bigframes.dataframe.DataFrame:
    if session is None:
        session = create_bigquery_session()

    # Since this may create a ReadLocalNode, the session we explicitly pass in
    # might not actually be used. Mock out the global session, too.
    monkeypatch.setattr(bigframes.core.global_session, "_global_session", session)
    bigframes.options.bigquery._session_started = True
    return bigframes.dataframe.DataFrame({"col": []}, session=session)


def create_polars_session() -> bigframes.Session:
    # TODO(tswast): Refactor to make helper available for all tests. Consider
    # providing a proper "local Session" for use by downstream developers.
    return TestSession()
