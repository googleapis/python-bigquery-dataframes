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

import dataclasses
import typing
from unittest import mock
import weakref

from google.cloud import bigquery

import bigframes.core
import bigframes.core.compile.sqlglot as sqlglot
import bigframes.dataframe
import bigframes.session.executor
import bigframes.session.metrics
import bigframes.session.temporary_storage
import bigframes.session.time


@dataclasses.dataclass
class SQLCompilerExecutor(bigframes.session.executor.Executor):
    """Executor for SQL compilation using sqlglot."""

    compiler = sqlglot

    def to_sql(
        self,
        array_value: bigframes.core.ArrayValue,
        offset_column: typing.Optional[str] = None,
        ordered: bool = True,
        enable_cache: bool = False,
    ) -> str:
        if offset_column:
            array_value, _ = array_value.promote_offsets()

        # Compared with BigQueryCachingExecutor, SQLCompilerExecutor skips
        # caching the subtree.
        return self.compiler.SQLGlotCompiler().compile(
            array_value.node, ordered=ordered
        )


class SQLCompilerSession(bigframes.session.Session):
    """Session for SQL compilation using sqlglot."""

    def __init__(self):
        # TODO: remove unused attributes.
        self._location = None  # type: ignore
        self._bq_kms_key_name = None  # type: ignore
        self._clients_provider = None  # type: ignore
        self.ibis_client = None  # type: ignore
        self._bq_connection = None  # type: ignore
        self._skip_bq_connection_check = True
        self._objects: list[
            weakref.ReferenceType[
                typing.Union[
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

        self._session_id: str = "sqlglot_unit_tests_session"
        self._executor = SQLCompilerExecutor()

        self._temp_storage_manager = self._mock_temp_storage_manager()  # type: ignore
        self._loader = bigframes.session.loader.GbqDataLoader(
            session=self,
            bqclient=self._mock_bq_client(),
            storage_manager=self._temp_storage_manager,
            write_client=None,  # type: ignore
            default_index_type=self._default_index_type,
            scan_index_uniqueness=self._strictly_ordered,
            force_total_order=self._strictly_ordered,
            metrics=self._metrics,
            # synced_clock=mock.create_autospec(
            #     bigframes.session.time.BigQuerySyncedClock
            # ),
        )

    def _mock_bq_client(self) -> bigquery.Client:
        mock_client = mock.create_autospec(bigquery.Client(project="project"))
        mock_table = mock.create_autospec(bigquery.Table("project.dataset.table"))
        type(mock_table).created = None
        type(mock_table).location = "US"
        type(mock_table).project = "project"
        type(mock_table).dataset_id = "dataset"
        type(mock_table).table_id = "table"
        type(mock_table).table_constraints = None
        mock_client.get_table.return_value = mock_table

        client = make_client()
        mock_client._connection = make_connection(
            {
                "jobReference": {
                    "projectId": "response-project",
                    "jobId": "response-job-id",
                    "location": "response-location",
                },
                "jobComplete": True,
                "queryId": "xyz",
                "schema": {
                    "fields": [
                        {"name": "full_name", "type": "STRING", "mode": "REQUIRED"},
                        {"name": "age", "type": "INT64", "mode": "NULLABLE"},
                    ],
                },
                "rows": [
                    {"f": [{"v": "Whillma Phlyntstone"}, {"v": "27"}]},
                    {"f": [{"v": "Bhetty Rhubble"}, {"v": "28"}]},
                    {"f": [{"v": "Phred Phlyntstone"}, {"v": "32"}]},
                    {"f": [{"v": "Bharney Rhubble"}, {"v": "33"}]},
                ],
                # Even though totalRows <= len(rows), we should use the presence of a
                # next page token to decide if there are any more pages.
                "totalRows": 2,
                "pageToken": "page-2",
            },
            # TODO(swast): This is a case where we can avoid a call to jobs.get,
            # but currently do so because the RowIterator might need the
            # destination table, since results aren't fully cached.
            {
                "jobReference": {
                    "projectId": "response-project",
                    "jobId": "response-job-id",
                    "location": "response-location",
                },
                "status": {"state": "DONE"},
            },
            {
                "rows": [
                    {"f": [{"v": "Pebbles Phlyntstone"}, {"v": "4"}]},
                    {"f": [{"v": "Bamm-Bamm Rhubble"}, {"v": "5"}]},
                    {"f": [{"v": "Joseph Rockhead"}, {"v": "32"}]},
                    {"f": [{"v": "Perry Masonry"}, {"v": "33"}]},
                ],
                "totalRows": 3,
                "pageToken": "page-3",
            },
            {
                "rows": [
                    {"f": [{"v": "Pearl Slaghoople"}, {"v": "53"}]},
                ],
                "totalRows": 4,
            },
        )

        mock_client.query_and_wait.return_value = {
            "jobReference": {
                "projectId": "response-project",
                "jobId": "abc",
                "location": "US",
            },
            "jobComplete": True,
            "queryId": "xyz",
            "schema": {
                "fields": [
                    {"name": "full_name", "type": "STRING", "mode": "REQUIRED"},
                    {"name": "age", "type": "INT64", "mode": "NULLABLE"},
                ],
            },
            "rows": [
                {"f": [{"v": "Whillma Phlyntstone"}, {"v": "27"}]},
                {"f": [{"v": "Bhetty Rhubble"}, {"v": "28"}]},
                {"f": [{"v": "Phred Phlyntstone"}, {"v": "32"}]},
                {"f": [{"v": "Bharney Rhubble"}, {"v": "33"}]},
            ],
            # Even though totalRows > len(rows), we should use the presence of a
            # next page token to decide if there are any more pages.
            "totalRows": 8,
        }
        # mock_client.query_and_wait = 
        #                 "SELECT CURRENT_TIMESTAMP() AS `current_timestamp`",
        #             )
        return mock_client

    def _mock_temp_storage_manager(
        self,
    ) -> bigframes.session.temporary_storage.TemporaryStorageManager:
        mock_temp_storage_manager = mock.create_autospec(
            bigframes.session.temporary_storage.TemporaryStorageManager
        )
        type(mock_temp_storage_manager).location = "US"
        return mock_temp_storage_manager

def make_connection(*responses):
    import google.cloud.bigquery._http
    from google.cloud.exceptions import NotFound

    mock_conn = mock.create_autospec(google.cloud.bigquery._http.Connection)
    mock_conn.user_agent = "testing 1.2.3"
    mock_conn.api_request.side_effect = list(responses) + [NotFound("miss")]
    mock_conn.API_BASE_URL = "https://bigquery.googleapis.com"
    mock_conn.get_api_base_url_for_mtls = mock.Mock(return_value=mock_conn.API_BASE_URL)
    return mock_conn