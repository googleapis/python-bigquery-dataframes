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

from typing import Optional, Sequence
import uuid

import google.cloud.bigquery as bigquery

import bigframes.session._io.bigquery as bf_io_bigquery


class SessionResourceManager:
    """
    Responsible for allocating and cleaning up temporary gbq tables used by a BigFrames session.
    """

    def __init__(
        self, bqclient: bigquery.Client, location: str, *, kms_key: Optional[str] = None
    ):
        self.bqclient = bqclient
        self.location = location
        self._kms_key = kms_key

    def create_temp_table(
        self, schema: Sequence[bigquery.SchemaField], cluster_cols: Sequence[str] = []
    ) -> bigquery.TableReference:
        # Can't set a table in _SESSION as destination via query job API, so we
        # run DDL, instead.

        table_ref = bigquery.TableReference(
            bigquery.DatasetReference(self.bqclient.project, "_SESSION"),
            uuid.uuid4().hex,
        )
        return bf_io_bigquery.create_temp_table(
            self.bqclient,
            table_ref,
            bq_session_id=self.get_next_free_session(),
            schema=schema,
            cluster_columns=list(cluster_cols),
            kms_key=self._kms_key,
        )

    # Alternatively, could reuse the same session, or lease from a pool of sessions
    def get_next_free_session(self) -> str:
        job_config = bigquery.QueryJobConfig(create_session=True)
        # Make sure the session is a new one, not one associated with another query.
        job_config.use_query_cache = False
        query_job = self.bqclient.query(
            "SELECT 1", job_config=job_config, location=self.location
        )
        query_job.result()  # blocks until finished
        return query_job.session_info.session_id

    def clean_up(self) -> None:
        ...