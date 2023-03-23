"""Session manages the connection to BigQuery."""

import re
import typing
from typing import Iterable, List, Optional, Tuple, Union

import google.auth.credentials
import google.cloud.bigquery as bigquery
import ibis
from ibis.backends.bigquery import Backend

from bigframes.core import BigFramesExpr
import bigframes.core.blocks as blocks
from bigframes.dataframe import DataFrame
import bigframes.version

_APPLICATION_NAME = f"bigframes/{bigframes.version.__version__}"


def _is_query(query_or_table: str) -> bool:
    """Determine if `query_or_table` is a table ID or a SQL string"""
    return re.search(r"\s", query_or_table.strip(), re.MULTILINE) is not None


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
        if context is not None:
            # TODO(chelsealin): Add the `location` parameter to ibis client.
            self.ibis_client = typing.cast(
                Backend,
                ibis.bigquery.connect(
                    project_id=context.project,
                    credentials=context.credentials,
                    application_name=_APPLICATION_NAME,
                ),
            )
        else:
            self.ibis_client = typing.cast(Backend, ibis.bigquery.connect())

        self.bqclient = self.ibis_client.client
        self._create_and_bind_bq_session()

    def _create_and_bind_bq_session(self):
        """Create a BQ session and bind the session id with clients to capture BQ activities:
        go/bigframes-transient-data"""
        job_config = bigquery.QueryJobConfig(create_session=True)
        query_job = self.bqclient.query("SELECT 1", job_config=job_config)
        self._session_id = query_job.session_info.session_id

        self.bqclient.default_query_job_config = bigquery.QueryJobConfig(
            connection_properties=[
                bigquery.ConnectionProperty("session_id", self._session_id)
            ]
        )
        self.bqclient.default_load_job_config = bigquery.LoadJobConfig(
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

    def read_gbq(
        self,
        query_or_table: str,
        *,
        col_order: Optional[List[str]] = None,
        index_cols: Union[Iterable[str], Tuple] = (),
    ) -> "DataFrame":
        """Loads DataFrame from Google BigQuery.

        Parameters
        ----------
        query_or_table : str
            BigQuery table name to be read, in the form `project.dataset.tablename` or
            `dataset.tablename`,  or a SQL string to be executed
        col_order : list(str), optional
            List of BigQuery column names in the desired order for results DataFrame.
        index_cols: list(str), optional
            List of column names to use as the index or multi-index

        Returns
        -------
        df: DataFrame
            DataFrame representing results of the query or table.
        """
        if _is_query(query_or_table):
            table_expression = self.ibis_client.sql(query_or_table)
        else:
            # TODO(swast): If a table ID, make sure we read from a snapshot to
            # better emulate pandas.read_gbq's point-in-time download. See:
            # https://cloud.google.com/bigquery/docs/time-travel#query_data_at_a_point_in_time
            table_ref = bigquery.table.TableReference.from_string(
                query_or_table, default_project=self.bqclient.project
            )
            table_expression = self.ibis_client.table(
                table_ref.table_id,
                database=f"{table_ref.project}.{table_ref.dataset_id}",
            )

        columns = None
        if col_order is not None:
            columns = tuple(
                table_expression[key] for key in col_order if key in table_expression
            )
            if len(columns) != len(col_order):
                raise ValueError("Column order does not match this table.")
        block = blocks.Block(BigFramesExpr(self, table_expression, columns), index_cols)
        return DataFrame(block)


def connect(context: Optional[Context] = None) -> Session:
    return Session(context)
