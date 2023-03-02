"""Session manages the connection to BigQuery."""

from typing import Optional

import google.auth.credentials
import google.cloud.bigquery as bigquery
import ibis

from bigframes.core import BigFramesExpr
import bigframes.core.blocks as blocks
from bigframes.dataframe import DataFrame


class Context:
    """Encapsulates configuration for working with an Session.

    Attributes:
      credentials: The OAuth2 Credentials to use for this client. If not passed
        falls back to the default inferred from the environment.
      project: Project ID for the project which the client acts on behalf of. Will
        be passed when creating a dataset / job. If not passed, falls back to the
        default inferred from the environment.
      location: Default location for jobs / datasets / tables.
    """

    def __init__(
        self,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
    ):
        self._credentials = credentials
        self._project = project
        self._location = location

    @property
    def credentials(self) -> Optional[google.auth.credentials.Credentials]:
        return self._credentials

    @property
    def project(self) -> Optional[str]:
        return self._project

    @property
    def location(self) -> Optional[str]:
        return self._location


class Session:
    """Establishes a BigQuery connection to capture a group of job activities related to
    DataFrames."""

    def __init__(self, context: Optional[Context] = None):
        # TODO(chelsealin): the ibis_client object has one more bq client, which takes additional
        # connection time on authentications etc, and does not have the session id assorted. A
        # change required in the ibis library to reuse the bq client or its _credential object.
        if context is not None:
            self.bqclient = bigquery.Client(
                credentials=context.credentials,
                project=context.project,
                location=context.location,
                default_query_job_config=bigquery.QueryJobConfig(),
            )
            self.ibis_client = ibis.bigquery.connect(
                project_id=context.project, credentials=context.credentials
            )
        else:
            self.bqclient = bigquery.Client()
            self.ibis_client = ibis.bigquery.connect(project_id=self.bqclient.project)

        self._create_and_bind_bq_session()

    def _create_and_bind_bq_session(self):
        """Create a BQ session and bind the session id with clients to capture BQ activities:
        go/bigframes-transient-data"""
        job_config = bigquery.QueryJobConfig(create_session=True)
        query_job = self.bqclient.query("SELECT 1", job_config=job_config)
        self._session_id = query_job.session_info.session_id

        # TODO(chelsealin): Don't call the private objects after sending a PR to BQ Python client
        # library.
        self.bqclient._default_query_job_config = bigquery.QueryJobConfig(
            connection_properties=[
                bigquery.ConnectionProperty("session_id", self._session_id)
            ]
        )

    def close(self):
        """Terminated the BQ session, otherwises the session will be terminated automatically after
        24 hours of inactivity or after 7 days."""
        if self._session_id is not None and self.bqclient is not None:
            abort_session_query = "CALL BQ.ABORT_SESSION('{}')".format(self._session_id)
            query_job = self.bqclient.query(abort_session_query)
            query_job.result()  # blocks until finished
            self._session_id = None

    def read_gbq(self, table: str) -> "DataFrame":
        """Loads data from Google BigQuery."""
        # TODO(swast): If a table ID, make sure we read from a snapshot to
        # better emulate pandas.read_gbq's point-in-time download. See:
        # https://cloud.google.com/bigquery/docs/time-travel#query_data_at_a_point_in_time
        table_ref = bigquery.table.TableReference.from_string(
            table, default_project=self.bqclient.project
        )
        table_expression = self.ibis_client.table(
            table_ref.table_id, database=f"{table_ref.project}.{table_ref.dataset_id}"
        )
        block = blocks.Block(BigFramesExpr(self, table_expression))
        return DataFrame(block)


def connect(context: Optional[Context] = None) -> Session:
    # TODO(swast): Start a BigQuery Session too.
    return Session(context)
