"""Session manages the connection to BigQuery."""

import re
import typing
from typing import Iterable, List, Optional, Tuple, Union
import uuid

import google.auth.credentials
import google.cloud.bigquery as bigquery
import ibis
from ibis.backends.bigquery import Backend
import pandas

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

        Args:
            query_or_table: a SQL string to be executed or a BigQuery table to be read. The
              table must be specified in the format of `project.dataset.tablename` or
              `dataset.tablename`.
            col_order: List of BigQuery column names in the desired order for results DataFrame.
            index_cols: List of column names to use as the index or multi-index

        Returns:
            A DataFrame representing results of the query or table.
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

    def read_pandas(self, pandas_dataframe: pandas.DataFrame) -> "DataFrame":
        """Loads DataFrame from a Pandas DataFrame.

        The Pandas DataFrame will be persisted as a temporary BigQuery table, which can be
        automatically recycled after the Session is closed.

        Args:
            pandas_dataframe: a Pandas DataFrame object to be loaded.

        Returns:
            A BigFrame DataFrame.
        """

        table_name = f"{uuid.uuid4().hex}"
        load_table_name = f"{self.bqclient.project}._SESSION.{table_name}"
        load_job = self.bqclient.load_table_from_dataframe(
            pandas_dataframe, load_table_name, job_config=bigquery.LoadJobConfig()
        )
        load_job.result()  # Wait for the job to complete

        # Both default indexes and unnamed non-default indexes are treated the same
        # and are not copied to BigQuery when load_table_from_dataframe executes
        if pandas_dataframe.index.name and pandas_dataframe.index.names:
            return self.read_gbq(
                f"SELECT * FROM `{table_name}`", index_cols=pandas_dataframe.index.names
            )
        else:
            return self.read_gbq(f"SELECT * FROM `{table_name}`")

    def read_csv(
        self,
        filepath: str,
        header: Optional[int] = None,
    ) -> "DataFrame":
        """Loads DataFrame from a comma-separated values (csv) file on GCS.

        The CSV file data will be persisted as a temporary BigQuery table, which can be
        automatically recycled after the Session is closed.

        Args:
            filepath: a string path including GS and local file.

            header:
                The number of rows at the top of a CSV file that BigQuery will skip when
                loading the data.
                - ``None``: Autodetect tries to detect headers in the first row. If they are
                not detected, the row is read as data. Otherwise data is read starting from
                the second row.
                - ``0``: Instructs autodetect that there are no headers and data should be
                read starting from the first row.
                - ``N > 0``: Autodetect skips N-1 rows and tries to detect headers in row N.
                If headers are not detected, row N is just skipped. Otherwise row N is used
                to extract column names for the detected schema.

        Returns:
            A BigFrame DataFrame.
        """
        # TODO(chelsealin): Supports more parameters defined at go/bigframes-io-api.
        # TODO(chelsealin): Supports to read local CSV file.
        if not filepath.startswith("gs://"):
            raise NotImplementedError(
                "Only Google Cloud Storage (gs://...) paths are supported."
            )

        table_name = f"{uuid.uuid4().hex}"
        dataset = bigquery.Dataset(
            bigquery.DatasetReference(self.bqclient.project, "_SESSION")
        )
        table = bigquery.Table(dataset.table(table_name))

        job_config = bigquery.LoadJobConfig()
        job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
        job_config.autodetect = True

        if header is not None:
            job_config.skip_leading_rows = header

        load_job = self.bqclient.load_table_from_uri(
            filepath, table, job_config=job_config
        )
        load_job.result()  # Wait for the job to complete

        return self.read_gbq(f"SELECT * FROM `{table_name}`")


def connect(context: Optional[Context] = None) -> Session:
    return Session(context)
