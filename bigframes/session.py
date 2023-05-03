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

"""Session manages the connection to BigQuery."""

from __future__ import annotations

import logging
import os
import re
import typing
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    MutableSequence,
    Optional,
    Tuple,
    Union,
)
import uuid

import google.api_core.exceptions
import google.auth.credentials
import google.cloud.bigquery as bigquery
import google.cloud.storage as storage  # type: ignore
import ibis
import ibis.backends.bigquery as ibis_bigquery
import ibis.expr.types as ibis_types
import numpy as np
import pandas
import pydata_google_auth

import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.indexes as indexes
from bigframes.core.ordering import OrderingColumnReference
import bigframes.dataframe as dataframe
import bigframes.ml.loader
from bigframes.remote_function import remote_function as biframes_rf
import bigframes.version

_ENV_DEFAULT_PROJECT = "GOOGLE_CLOUD_PROJECT"
_APPLICATION_NAME = f"bigframes/{bigframes.version.__version__}"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

logger = logging.getLogger(__name__)


def _is_query(query_or_table: str) -> bool:
    """Determine if `query_or_table` is a table ID or a SQL string"""
    return re.search(r"\s", query_or_table.strip(), re.MULTILINE) is not None


# TODO(shobs): Remove it after the same is available via pydata-google-auth
# after https://github.com/pydata/pydata-google-auth/pull/68 is merged
def _ensure_application_default_credentials_in_colab_environment():
    # This is a special handling for google colab environment where we want to
    # use the colab specific authentication flow
    # https://github.com/googlecolab/colabtools/blob/3c8772efd332289e1c6d1204826b0915d22b5b95/google/colab/auth.py#L209
    try:
        from google.colab import auth

        auth.authenticate_user()
    except (ModuleNotFoundError, ImportError):
        pass


class Context:
    """Encapsulates configuration for working with an Session.

    Attributes:
      credentials: The OAuth2 Credentials to use for this client. If not passed
        falls back to the default inferred from the environment.
      project: Project ID for the project which the client acts on behalf of. Will
        be passed when creating a dataset / job. If not passed, falls back to the
        default inferred from the environment.
      location: Default location for jobs / datasets / tables.
      bigquery_connection: Name of the BigQuery connection for the purpose of
        remote udfs. It should be either pre created in `location`, or the user
        should have privilege to create one.
    """

    def __init__(
        self,
        credentials: Optional[google.auth.credentials.Credentials] = None,
        project: Optional[str] = None,
        location: Optional[str] = None,
        bigquery_connection: Optional[str] = None,
    ):
        self._credentials = credentials
        self._project = project
        self._location = location
        self._bigquery_connection = bigquery_connection

    @property
    def credentials(self) -> Optional[google.auth.credentials.Credentials]:
        return self._credentials

    @credentials.setter
    def credentials(self, value: Optional[google.auth.credentials.Credentials]):
        self._credentials = value

    @property
    def project(self) -> Optional[str]:
        return self._project

    @project.setter
    def project(self, value: Optional[str]):
        self._project = value

    @property
    def location(self) -> Optional[str]:
        return self._location

    @location.setter
    def location(self, value: Optional[str]):
        self._location = value

    @property
    def bigquery_connection(self) -> Optional[str]:
        return self._bigquery_connection

    @bigquery_connection.setter
    def bigquery_connection(self, value: Optional[str]):
        self._bigquery_connection = value


class Session:
    """Establishes a BigQuery connection to capture a group of job activities related to
    DataFrames."""

    def __init__(self, context: Optional[Context] = None):
        if context is None:
            context = Context()

        # We want to initiate auth via a non-local web server which particularly
        # helps in a cloud notebook environment where the machine running the
        # notebook UI and the VM running the notebook runtime are not the same.
        if context.credentials is None:
            _ensure_application_default_credentials_in_colab_environment()
            # TODO(shobs, b/278903498): Use BigFrames own client id and secret
            context.credentials, pydata_default_project = pydata_google_auth.default(
                _SCOPES, use_local_webserver=False
            )
            if not context.project:
                context.project = pydata_default_project

        # If there is no project set yet, try to set it from the environment
        if not context.project:
            context.project = os.environ.get(_ENV_DEFAULT_PROJECT, context.project)

        # TODO(chelsealin): Add the `location` parameter to ibis client.
        self.ibis_client = typing.cast(
            ibis_bigquery.Backend,
            ibis.bigquery.connect(
                project_id=context.project,
                credentials=context.credentials,
                application_name=_APPLICATION_NAME,
            ),
        )

        self.bqclient = self.ibis_client.client
        # TODO(swast): Get location from the environment.
        self._location = (
            "US" if context is None or context.location is None else context.location
        )
        self._create_and_bind_bq_session()
        self._bigquery_connection = context.bigquery_connection

    @property
    def _session_dataset_id(self):
        """A dataset for storing temporary objects local to the session
        This is a workaround for BQML models and remote functions that do not
        yet support session-temporary instances."""
        return self._session_dataset.dataset_id

    def _create_and_bind_bq_session(self):
        """Create a BQ session and bind the session id with clients to capture BQ activities:
        go/bigframes-transient-data"""
        job_config = bigquery.QueryJobConfig(create_session=True)
        query_job = self.bqclient.query(
            "SELECT 1", job_config=job_config, location=self._location
        )
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

        # Dataset for storing BQML models and remote functions, which don't yet
        # support proper session temporary storage yet
        self._session_dataset = bigquery.Dataset(
            f"{self.bqclient.project}.bigframes_temp_{self._location.lower().replace('-', '_')}"
        )
        self._session_dataset.location = self._location
        self._session_dataset.default_table_expiration_ms = 24 * 60 * 60 * 1000

        # TODO: handle case when the dataset does not exist and the user does
        # not have permission to create one (bigquery.datasets.create IAM)
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
            index_col_name = indexes.INDEX_COLUMN_ID.format(0)
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

        df = dataframe.DataFrame(block.index)
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

        # Column values will be loaded as null if the column name has spaces.
        # https://github.com/googleapis/python-bigquery/issues/1566
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
            ordering_id_column=OrderingColumnReference(ordering_col), is_sequential=True
        )
        return self._read_gbq_with_ordering(
            f"SELECT * FROM `{load_table_destination.table_id}`",
            index_cols=index_cols,
            ordering=ordering,
        )

    def read_csv(
        self,
        filepath_or_buffer: str,
        *,
        header: Optional[int] = 0,
        engine: Optional[
            Literal["c", "python", "pyarrow", "python-fwf", "bigquery"]
        ] = None,
        dtype: Optional[Dict] = None,
        names: Optional[
            Union[MutableSequence[Any], np.ndarray[Any, Any], Tuple[Any, ...], range]
        ] = None,
        **kwargs,
    ) -> dataframe.DataFrame:
        """Loads DataFrame from comma-separated values (csv) file locally or from GCS.

        The CSV file data will be persisted as a temporary BigQuery table, which can be
        automatically recycled after the Session is closed.

        Note: using the `bigquery` engine will not guarantee the same ordering as the
        file in the resulting dataframe.

        Args:
            filepath_or_buffer: a string path including GS and local file.

            header: row number to use as the column names.
                - ``None``: Instructs autodetect that there are no headers and data should be
                read starting from the first row.
                - ``0``: If using engine="bigquery", Autodetect tries to detect headers in the
                first row. If they are not detected, the row is read as data. Otherwise data
                is read starting from the second row. When using default engine, pandas assumes
                the first row contains column names unless the `names` argument is specified.
                If `names` is provided, then the first row is ignored, second row is read as
                data, and column names are inferred from `names`.
                - ``N > 0``: If using engine="bigquery", Autodetect skips N rows and tries
                to detect headers in row N+1. If headers are not detected, row N+1 is just
                skipped. Otherwise row N+1 is used to extract column names for the detected
                schema. When using default engine, pandas will skip N rows and assumes row N+1
                contains column names unless the `names` argument is specified. If `names` is
                provided, row N+1 will be ignored, row N+2 will be read as data, and column
                names are inferred from `names`.

            engine: type of engine to use. If "bigquery" is specified, then BigQuery's load
                API will be used. Otherwise, the engine will be passed to pandas.read_csv.

            dtype: data type for data or columns. Only to be used with default engine.

            names: a list of column names to use. If the file contains a header row and you
                want to pass this parameter, then `header=0` should be passed as well so the
                first (header) row is ignored. Only to be used with default engine.

            **kwargs: keyword arguments.


        Returns:
            A BigFrame DataFrame.
        """
        # TODO(chelsealin): Supports more parameters defined at go/bigframes-io-api.
        table = bigquery.Table(self._create_session_table())

        if engine is not None and engine == "bigquery":
            if any(param is not None for param in (dtype, names)):
                not_supported = ("dtype", "names")
                raise NotImplementedError(
                    f"BigQuery engine does not support these arguments: {not_supported}"
                )

            if not isinstance(filepath_or_buffer, str):
                raise NotImplementedError("BigQuery engine does not support buffers.")

            job_config = bigquery.LoadJobConfig()
            job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
            job_config.source_format = bigquery.SourceFormat.CSV
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            job_config.autodetect = True

            # We want to match pandas behavior. If header is 0, no rows should be skipped, so we
            # do not need to set `skip_leading_rows`. If header is None, then there is no header.
            # Setting skip_leading_rows to 0 does that. If header=N and N>0, we want to skip N rows.
            # `skip_leading_rows` skips N-1 rows, so we set it to header+1.
            if header is not None and header > 0:
                job_config.skip_leading_rows = header + 1
            elif header is None:
                job_config.skip_leading_rows = 0

            if filepath_or_buffer.startswith("gs://"):
                load_job = self.bqclient.load_table_from_uri(
                    filepath_or_buffer, table, job_config=job_config
                )
            else:
                with open(filepath_or_buffer, "rb") as source_file:
                    load_job = self.bqclient.load_table_from_file(
                        source_file, table, job_config=job_config
                    )
            load_job.result()  # Wait for the job to complete
            return self.read_gbq(f"SELECT * FROM `{table.table_id}`")
        else:
            if any(arg in kwargs for arg in ["chunksize", "iterator"]):
                raise NotImplementedError(
                    "'chunksize' and 'iterator' arguments are not supported."
                )

            self._check_file_size(filepath_or_buffer)
            pandas_df = pandas.read_csv(
                filepath_or_buffer,
                header=header,
                engine=engine,
                dtype=dtype,
                names=names,
                **kwargs,
            )
            return self.read_pandas(pandas_df)

    def _check_file_size(self, filepath: str):
        max_size = 1024 * 1024 * 1024  # 1 GB in bytes
        if filepath.startswith("gs://"):  # GCS file path
            client = storage.Client()
            bucket_name, blob_name = filepath.split("/", 3)[2:]
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.reload()
            file_size = blob.size
        else:  # local file path
            file_size = os.path.getsize(filepath)

        if file_size > max_size:
            # Convert to GB
            file_size = round(file_size / (1024**3), 1)
            max_size = int(max_size / 1024**3)
            logger.warning(
                f"File size {file_size}GB exceeds {max_size}GB. "
                "It is recommended to use engine='bigquery' "
                "for large files to avoid loading the file into local memory."
            )

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

    def remote_function(
        self,
        input_types: List[ibis.expr.datatypes.core.DataType],
        output_type: ibis.expr.datatypes.core.DataType,
        dataset: Optional[str] = None,
        bigquery_connection: Optional[str] = None,
        reuse: bool = True,
    ):
        """Create a remote function from a user defined function."""

        return biframes_rf(
            input_types,
            output_type,
            session=self,
            dataset=dataset,
            bigquery_connection=bigquery_connection,
            reuse=reuse,
        )


def connect(context: Optional[Context] = None) -> Session:
    return Session(context)
