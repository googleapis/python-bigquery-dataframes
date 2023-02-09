"""Session manages the connection to BigQuery."""

from typing import Optional

import google.auth.credentials
import google.cloud.bigquery
import ibis
from google.cloud.bigquery.table import TableReference

from bigframes.core import BigFramesExpr
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
    """Establishes a BigQuery connection to capture a group of job activities related to DataFrames."""

    def __init__(self, context: Optional[Context] = None):
        # TODO(chelsealin): create a BigQuery Session at connect time.
        if context is not None:
            self.bqclient = google.cloud.bigquery.Client(
                credentials=context.credentials,
                project=context.project,
                location=context.location,
            )
            self.ibis_client = ibis.bigquery.connect(
                project_id=context.project, credentials=context.credentials
            )
        else:
            self.bqclient = google.cloud.bigquery.Client()
            self.ibis_client = ibis.bigquery.connect(project_id=self.bqclient.project)

    def read_gbq(self, table: str) -> "DataFrame":
        """Loads data from Google BigQuery."""
        # TODO(swast): If a table ID, make sure we read from a snapshot to
        # better emulate pandas.read_gbq.
        table_ref = TableReference.from_string(
            table, default_project=self.bqclient.project
        )
        table_expression = self.ibis_client.table(
            table_ref.table_id, database=f"{table_ref.project}.{table_ref.dataset_id}"
        )
        return DataFrame(BigFramesExpr(self, table_expression))


def connect(context: Optional[Context] = None) -> Session:
    # TODO(swast): Start a BigQuery Session too.
    return Session(context)
