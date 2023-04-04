"""Session manages the connection to BigQuery."""

from __future__ import annotations

import re
import typing
from typing import Iterable, List, Optional, Tuple, Union
import uuid

import google.api_core.exceptions
import google.auth.credentials
import google.cloud.bigquery as bigquery
import ibis
import ibis.backends.bigquery as ibis_bigquery
import ibis.expr.types as ibis_types
import numpy as np
import pandas

import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.indexes as indexes
import bigframes.dataframe as dataframe
import bigframes.ml.loader
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
                ibis_bigquery.Backend,
                ibis.bigquery.connect(
                    project_id=context.project,
                    credentials=context.credentials,
                    application_name=_APPLICATION_NAME,
                ),
            )
        else:
            self.ibis_client = typing.cast(
                ibis_bigquery.Backend, ibis.bigquery.connect()
            )

        self.bqclient = self.ibis_client.client
        self._create_and_bind_bq_session()

    @property
    def _session_dataset_id(self):
        """A dataset for storing temporary objects local to the session
        This is a workaround for BQML models (and remote functions?) that do not yet support
        session-temporary instances."""
        return self._session_dataset.dataset_id

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

        # Dataset for storing BQML models and remote functions, which don't yet support proper
        # session temporary storage yet
        # TODO(b/276793359): set location dynamically per go/bigframes-transient-data
        self._session_dataset = bigquery.Dataset(
            f"{self.bqclient.project}.bigframes_temp_us"
        )
        self._session_dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000

        # TODO: handle case when the dataset does not exist and the user does not have permission
        # to create one (bigquery.datasets.create IAM)
        self.bqclient.create_dataset(self._session_dataset, exists_ok=True)

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
        col_order: Optional[Iterable[str]] = None,
        index_cols: Iterable[str] = (),
    ) -> dataframe.DataFrame:
        """Loads DataFrame from Google BigQuery.

        Args:
            query_or_table: a SQL string to be executed or a BigQuery table to be read. The
              table must be specified in the format of `project.dataset.tablename` or
              `dataset.tablename`.
            col_order: List of BigQuery column names in the desired order for results DataFrame.
            index_cols: List of column names to use as the index or multi-index.

        Returns:
            A DataFrame representing results of the query or table.
        """
        return self._read_gbq_with_ordering(
            query_or_table=query_or_table, col_order=col_order, index_cols=index_cols
        )

    def _read_gbq_with_ordering(
        self,
        query_or_table: str,
        *,
        col_order: Optional[Iterable[str]] = None,
        index_cols: Union[Iterable[str], Tuple] = (),
        ordering: Optional[core.ExpressionOrdering] = None,
    ) -> dataframe.DataFrame:
        """Internal helper method that loads DataFrame from Google BigQuery given an optional ordering column.

        Args:
            query_or_table: a SQL string to be executed or a BigQuery table to be read. The
              table must be specified in the format of `project.dataset.tablename` or
              `dataset.tablename`.
            col_order: List of BigQuery column names in the desired order for results DataFrame.
            index_cols: List of column names to use as the index or multi-index.
            ordering_col: Column name to be used for ordering.

        Returns:
            A DataFrame representing results of the query or table.
        """
        index_keys = list(index_cols)
        if len(index_keys) > 1:
            raise NotImplementedError("MultiIndex not supported.")

        if _is_query(query_or_table):
            table_expression = self.ibis_client.sql(query_or_table)
        else:
            # TODO(swast): Can we re-use the temp table from other reads in the
            # session, if the original table wasn't modified?
            table_ref = bigquery.table.TableReference.from_string(
                query_or_table, default_project=self.bqclient.project
            )
            table_expression = self.ibis_client.table(
                table_ref.table_id,
                database=f"{table_ref.project}.{table_ref.dataset_id}",
            )

        if index_keys:
            # TODO(swast): Support MultiIndex.
            index_col_name = index_keys[0]
            index_col = table_expression[index_col_name]
            index_name = index_keys[0]
        else:
            index_col_name = indexes.INDEX_COLUMN_NAME.format(0)
            index_name = None

            if ordering is not None and ordering.ordering_id:
                # Use the sequential ordering as the index instead of creating
                # a new one, if available.
                index_col = table_expression[ordering.ordering_id].name(index_col_name)
            else:
                # Add an arbitrary sequential index and materialize the table
                # because row_number() could refer to different rows depending
                # on how the rows in the DataFrame are filtered / dropped.
                index_col = ibis.row_number().name(index_col_name)
                table_expression = table_expression.mutate(
                    **{index_col_name: index_col}
                )
                table_expression = self._query_to_session_table(
                    table_expression.compile()
                )
                index_col = table_expression[index_col_name]

        if col_order is None:
            if ordering is not None and ordering.ordering_id:
                non_value_cols = {index_col_name, ordering.ordering_id}
            else:
                non_value_cols = {index_col_name}
            column_keys = [
                key for key in table_expression.columns if key not in non_value_cols
            ]
        else:
            column_keys = list(col_order)
        return self._read_ibis(
            table_expression, index_col, index_name, column_keys, ordering=ordering
        )

    def _read_ibis(
        self,
        table_expression: ibis_types.Table,
        index_col: ibis_types.Value,
        index_name: Optional[str],
        column_keys: List[str],
        ordering: Optional[core.ExpressionOrdering] = None,
    ):
        """Turns a table expression (plus index column) into a DataFrame."""
        meta_columns = None
        if ordering is not None:
            meta_columns = (table_expression[ordering.ordering_id],)

        columns = [index_col]
        for key in column_keys:
            if key not in table_expression.columns:
                raise ValueError(f"Column '{key}' not found in this table.")
            columns.append(table_expression[key])

        block = blocks.Block(
            core.BigFramesExpr(self, table_expression, columns, meta_columns, ordering),
            [index_col.get_name()],
        )

        df = dataframe.DataFrame(block)
        df.index.name = index_name
        return df

    def read_gbq_model(self, model_name: str):
        """Loads a BQML model from Google BigQuery.

        Args:
            model_name : the model's name in BigQuery in the format
            `project_id.dataset_id.model_id`, or just `dataset_id.model_id`
            to load from the default project.

        Returns:
            A bigframes.ml Model wrapping the model
        """
        model_ref = bigquery.ModelReference.from_string(
            model_name, default_project=self.bqclient.project
        )
        model = self.bqclient.get_model(model_ref)
        return bigframes.ml.loader.from_bq(self, model)

    def read_pandas(self, pandas_dataframe: pandas.DataFrame) -> dataframe.DataFrame:
        """Loads DataFrame from a Pandas DataFrame.

        The Pandas DataFrame will be persisted as a temporary BigQuery table, which can be
        automatically recycled after the Session is closed.

        Args:
            pandas_dataframe: a Pandas DataFrame object to be loaded.

        Returns:
            A BigFrame DataFrame.
        """
        # Add order column to pandas DataFrame to preserve order in BigQuery
        ordering_col = "rowid"
        columns = frozenset(pandas_dataframe.columns)
        suffix = 2
        while ordering_col in columns:
            ordering_col = f"rowid_{suffix}"
            suffix += 1

        pandas_dataframe_copy = pandas_dataframe.copy()
        pandas_dataframe_copy[ordering_col] = np.arange(pandas_dataframe_copy.shape[0])

        load_table_destination = self._create_session_table()
        load_job = self.bqclient.load_table_from_dataframe(
            pandas_dataframe_copy,
            load_table_destination,
            job_config=bigquery.LoadJobConfig(),
        )
        load_job.result()  # Wait for the job to complete

        # Both default indexes and unnamed non-default indexes are treated the same
        # and are not copied to BigQuery when load_table_from_dataframe executes
        index_cols = filter(
            lambda name: name is not None, pandas_dataframe_copy.index.names
        )
        ordering = core.ExpressionOrdering(
            ordering_id_column=ordering_col, is_sequential=True, ascending=True
        )
        return self._read_gbq_with_ordering(
            f"SELECT * FROM `{load_table_destination.table_id}`",
            index_cols=index_cols,
            ordering=ordering,
        )

    def read_csv(
        self,
        filepath: str,
        header: Optional[int] = None,
    ) -> dataframe.DataFrame:
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

        table = bigquery.Table(self._create_session_table())

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

        return self.read_gbq(f"SELECT * FROM `{table.table_id}`")

    def _create_session_table(self) -> bigquery.TableReference:
        table_name = f"{uuid.uuid4().hex}"
        dataset = bigquery.Dataset(
            bigquery.DatasetReference(self.bqclient.project, "_SESSION")
        )
        return dataset.table(table_name)

    def _query_to_session_table(self, query_text: str) -> ibis_types.Table:
        table = self._create_session_table()
        # TODO(swast): Can't set a table in _SESSION as destination, so we run
        # DDL, instead.
        # TODO(swast): This might not support multi-statement SQL queries.
        ddl_text = f"CREATE TEMPORARY TABLE `{table.table_id}` AS {query_text}"
        query_job = self.bqclient.query(ddl_text)
        try:
            query_job.result()  # Wait for the job to complete
        except google.api_core.exceptions.Conflict:
            # Allow query retry to succeed.
            pass
        return self.ibis_client.sql(f"SELECT * FROM `{table.table_id}`")


def connect(context: Optional[Context] = None) -> Session:
    return Session(context)
