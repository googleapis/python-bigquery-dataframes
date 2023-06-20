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
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import uuid

import google.api_core.client_info
import google.api_core.client_options
import google.api_core.exceptions
import google.api_core.gapic_v1.client_info
import google.auth.credentials
import google.cloud.bigquery as bigquery
import google.cloud.bigquery_connection_v1
import google.cloud.bigquery_storage_v1
import google.cloud.storage as storage  # type: ignore
import ibis
import ibis.backends.bigquery as ibis_bigquery
import ibis.expr.types as ibis_types
import numpy as np
import pandas
import pydata_google_auth

import bigframes._config.bigquery_options as bigquery_options
import bigframes.core as core
import bigframes.core.blocks as blocks
import bigframes.core.indexes as indexes
from bigframes.core.ordering import OrderingColumnReference
import bigframes.dataframe as dataframe
import bigframes.ml.loader
from bigframes.remote_function import remote_function as biframes_rf
import bigframes.version
import third_party.bigframes_vendored.pandas.io.gbq as third_party_pandas_gbq
import third_party.bigframes_vendored.pandas.io.parsers.readers as third_party_pandas_readers

_ENV_DEFAULT_PROJECT = "GOOGLE_CLOUD_PROJECT"
_APPLICATION_NAME = f"bigframes/{bigframes.version.__version__}"
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# BigQuery is a REST API, which requires the protocol as part of the URL.
_BIGQUERY_REGIONAL_ENDPOINT = "https://{location}-bigquery.googleapis.com"

# BigQuery Connection and Storage are gRPC APIs, which don't support the
# https:// protocol in the API endpoint URL.
_BIGQUERYCONNECTION_REGIONAL_ENDPOINT = "{location}-bigqueryconnection.googleapis.com"
_BIGQUERYSTORAGE_REGIONAL_ENDPOINT = "{location}-bigquerystorage.googleapis.com"

# TODO(swast): Need to connect to regional endpoints when performing remote
# functions operations (BQ Connection API, Cloud Run / Cloud Functions).

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
    except Exception:
        # We are catching a broad exception class here because we want to be
        # agnostic to anything that could internally go wrong in the google
        # colab auth. Some of the known exception we want to pass on are:
        #
        # ModuleNotFoundError: No module named 'google.colab'
        # ImportError: cannot import name 'auth' from 'google.cloud'
        # MessageError: Error: credential propagation was unsuccessful
        #
        # The MessageError happens on Vertex Colab when it fails to resolve auth
        # from the Compute Engine Metadata server.
        pass


class Session(
    third_party_pandas_gbq.GBQIOMixin, third_party_pandas_readers.ReaderIOMixin
):
    """Establishes a BigQuery connection to capture a group of job activities related to
    DataFrames."""

    def __init__(self, context: Optional[bigquery_options.BigQueryOptions] = None):
        if context is None:
            context = bigquery_options.BigQueryOptions()

        # We want to initiate auth via a non-local web server which particularly
        # helps in a cloud notebook environment where the machine running the
        # notebook UI and the VM running the notebook runtime are not the same.
        if context.credentials is not None:
            credentials = context.credentials
            credentials_project = None
        else:
            _ensure_application_default_credentials_in_colab_environment()
            # TODO(shobs, b/278903498): Use BigFrames own client id and secret
            credentials, credentials_project = pydata_google_auth.default(
                _SCOPES, use_local_webserver=False
            )

        project = context.project

        # If there is no project set yet, try to set it from the environment
        if project is None:
            # Prefer the project defined by environment variable, but fallback
            # to the project associated with credentials (if available).
            project = os.getenv(_ENV_DEFAULT_PROJECT) or typing.cast(
                Optional[str], credentials_project
            )

        # TODO(swast): Get location from the environment.
        self._location = (
            "US" if context is None or context.location is None else context.location
        )
        self._create_bq_clients(
            project=project,
            location=self._location,
            use_regional_endpoints=context.use_regional_endpoints,
            credentials=credentials,
        )
        self._create_and_bind_bq_session()
        self.ibis_client = typing.cast(
            ibis_bigquery.Backend,
            ibis.bigquery.connect(
                project_id=context.project,
                client=self.bqclient,
                storage_client=self.bqstorageclient,
            ),
        )

        self._remote_udf_connection = context.remote_udf_connection

        # Now that we're starting the session, don't allow the options to be
        # changed.
        context._session_started = True

    @property
    def _session_dataset_id(self):
        """A dataset for storing temporary objects local to the session
        This is a workaround for BQML models and remote functions that do not
        yet support session-temporary instances."""
        return self._session_dataset.dataset_id

    def _create_bq_clients(
        self,
        *,
        project: Optional[str],
        location: str,
        use_regional_endpoints: bool,
        credentials: google.auth.credentials.Credentials,
    ):
        """Create and initialize BigQuery client objects."""

        if use_regional_endpoints:
            bq_options = google.api_core.client_options.ClientOptions(
                api_endpoint=_BIGQUERY_REGIONAL_ENDPOINT.format(location=location),
            )
            bqstorage_options = google.api_core.client_options.ClientOptions(
                api_endpoint=_BIGQUERYSTORAGE_REGIONAL_ENDPOINT.format(
                    location=location
                )
            )
            bqconnection_options = google.api_core.client_options.ClientOptions(
                api_endpoint=_BIGQUERYCONNECTION_REGIONAL_ENDPOINT.format(
                    location=location
                )
            )
        else:
            bq_options = None
            bqstorage_options = None
            bqconnection_options = None

        location = location.lower()
        bq_info = google.api_core.client_info.ClientInfo(user_agent=_APPLICATION_NAME)
        self.bqclient = bigquery.Client(
            client_info=bq_info,
            client_options=bq_options,
            credentials=credentials,
            project=project,
        )

        bqconnection_info = google.api_core.gapic_v1.client_info.ClientInfo(
            user_agent=_APPLICATION_NAME
        )
        self.bqconnectionclient = (
            google.cloud.bigquery_connection_v1.ConnectionServiceClient(
                client_info=bqconnection_info,
                client_options=bqconnection_options,
                credentials=credentials,
            )
        )

        bqstorage_info = google.api_core.gapic_v1.client_info.ClientInfo(
            user_agent=_APPLICATION_NAME
        )
        self.bqstorageclient = google.cloud.bigquery_storage_v1.BigQueryReadClient(
            client_info=bqstorage_info,
            client_options=bqstorage_options,
            credentials=credentials,
        )

    def _create_and_bind_bq_session(self):
        """Create a BQ session and bind the session id with clients to capture BQ activities:
        go/bigframes-transient-data"""
        job_config = bigquery.QueryJobConfig(create_session=True)
        query_job = self.bqclient.query(
            "SELECT 1", job_config=job_config, location=self._location
        )
        query_job.result()  # blocks until finished
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
        query: str,
        *,
        index_col: Iterable[str] | str = (),
        col_order: Iterable[str] = (),
        max_results: Optional[int] = None,
    ) -> dataframe.DataFrame:
        # TODO(b/281571214): Generate prompt to show the progress of read_gbq.
        if _is_query(query):
            table_expression = self.ibis_client.sql(query)
        else:
            # TODO(swast): Can we re-use the temp table from other reads in the
            # session, if the original table wasn't modified?
            table_ref = bigquery.table.TableReference.from_string(
                query, default_project=self.bqclient.project
            )
            table_expression = self.ibis_client.table(
                table_ref.table_id,
                database=f"{table_ref.project}.{table_ref.dataset_id}",
            )

        for key in col_order:
            if key not in table_expression.columns:
                raise ValueError(
                    f"Column '{key}' of `col_order` not found in this table."
                )

        if isinstance(index_col, str):
            index_cols: Iterable[str] = [index_col]
        else:
            index_cols = index_col

        for key in index_cols:
            if key not in table_expression.columns:
                raise ValueError(
                    f"Column `{key}` of `index_col` not found in this table."
                )

        if max_results is not None:
            if max_results <= 0:
                raise ValueError("`max_results` should be a positive number.")
            table_expression = table_expression.limit(max_results)

        return self._read_gbq_with_ordering(
            table_expression=table_expression,
            col_order=col_order,
            index_cols=index_cols,
        )

    def _read_gbq_with_ordering(
        self,
        table_expression: ibis_types.Table,
        *,
        col_order: Iterable[str] = (),
        index_cols: Union[Iterable[str], Tuple] = (),
        ordering: Optional[core.ExpressionOrdering] = None,
    ) -> dataframe.DataFrame:
        """Internal helper method that loads DataFrame from Google BigQuery given an optional ordering column.

        Args:
            table_expression: an ibis table expression to be executed in BigQuery.
            col_order: List of BigQuery column names in the desired order for results DataFrame.
            index_cols: List of column names to use as the index or multi-index.
            ordering: Column name to be used for ordering. If not supplied, a default ordering is generated.

        Returns:
            A DataFrame representing results of the query or table.
        """
        index_keys = list(index_cols)
        if len(index_keys) > 1:
            raise NotImplementedError("MultiIndex not supported.")

        # Logic:
        # no ordering, no index -> create sequential order, use for both ordering and index
        # no ordering, index -> create sequential order, ordered by index, use for both ordering and index
        # sequential ordering, no index -> use ordering as default index
        # non-sequential ordering, no index -> NotImplementedException
        # sequential ordering, index -> use ordering as ordering, index as index

        # This code block ensures the existence of a sequential ordering column
        if ordering is None:
            # Rows are not ordered, we need to generate a default ordering and materialize it
            default_ordering_name = core.ORDER_ID_COLUMN
            default_ordering_col = ibis.row_number().name(default_ordering_name)
            table_expression = table_expression.mutate(
                **{default_ordering_name: default_ordering_col}
            )
            table_expression = self._query_to_session_table(table_expression.compile())
            ordering_reference = core.OrderingColumnReference(default_ordering_name)
            ordering = core.ExpressionOrdering(
                ordering_id_column=ordering_reference, is_sequential=True
            )
        elif not ordering.order_id_defined or not ordering.is_sequential:
            raise NotImplementedError(
                "Only sequential order by id column is supported for read_gbq"
            )
        else:
            default_ordering_name = typing.cast(str, ordering.ordering_id)

        if index_keys:
            # TODO(swast): Support MultiIndex.
            index_col_name = index_id = index_keys[0]
            index_col = table_expression[index_id]
        else:
            index_col_name = None
            # Make sure we have a separate "copy" of the ordering ID to use as
            # the index, because we assume the ordering ID is a hidden column
            # in BigFramesExpr, but the index column appears as a "column".
            index_id = indexes.INDEX_COLUMN_ID.format(0)
            index_col = table_expression[default_ordering_name].name(index_id)
            table_expression = table_expression.mutate(**{index_id: index_col})

        column_keys = list(col_order)
        if len(column_keys) == 0:
            column_keys = [
                key
                for key in table_expression.columns
                if key not in {index_id, ordering.ordering_id}
            ]
        return self._read_ibis(
            table_expression, index_col, index_col_name, column_keys, ordering=ordering
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
        hidden_ordering_columns = None
        if ordering is not None:
            hidden_ordering_columns = (table_expression[ordering.ordering_id],)

        columns = [index_col]
        for key in column_keys:
            if key not in table_expression.columns:
                raise ValueError(f"Column '{key}' not found in this table.")
            columns.append(table_expression[key])

        block = blocks.Block(
            core.BigFramesExpr(
                self, table_expression, columns, hidden_ordering_columns, ordering
            ),
            [index_col.get_name()],
            index_labels=[index_name],
        )

        return dataframe.DataFrame(block)

    def read_gbq_model(self, model_name: str):
        """Loads a BQML model from Google BigQuery.

        Args:
            model_name : the model's name in BigQuery in the format
            `project_id.dataset_id.model_id`, or just `dataset_id.model_id`
            to load from the default project.

        Returns:
            A bigframes.ml Model wrapping the model.
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

        # Specify the datetime dtypes, which is auto-detected as timestamp types.
        schema = []
        for column, dtype in zip(pandas_dataframe.columns, pandas_dataframe.dtypes):
            if dtype == "timestamp[us][pyarrow]":
                schema.append(
                    bigquery.SchemaField(column, bigquery.enums.SqlTypeNames.DATETIME)
                )

        # Column values will be loaded as null if the column name has spaces.
        # https://github.com/googleapis/python-bigquery/issues/1566
        load_table_destination = self._create_session_table()
        load_job = self.bqclient.load_table_from_dataframe(
            pandas_dataframe_copy,
            load_table_destination,
            job_config=bigquery.LoadJobConfig(schema=schema),
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
        table_expression = self.ibis_client.sql(
            f"SELECT * FROM `{load_table_destination.table_id}`"
        )
        return self._read_gbq_with_ordering(
            table_expression=table_expression,
            index_cols=index_cols,
            ordering=ordering,
        )

    def read_csv(
        self,
        filepath_or_buffer: str,
        *,
        sep: Optional[str] = ",",
        header: Optional[int] = 0,
        names: Optional[
            Union[MutableSequence[Any], np.ndarray[Any, Any], Tuple[Any, ...], range]
        ] = None,
        index_col: Optional[
            Union[int, str, Sequence[Union[str, int]], Literal[False]]
        ] = None,
        usecols: Optional[
            Union[
                MutableSequence[str],
                Tuple[str, ...],
                Sequence[int],
                pandas.Series,
                pandas.Index,
                np.ndarray[Any, Any],
                Callable[[Any], bool],
            ]
        ] = None,
        dtype: Optional[Dict] = None,
        engine: Optional[
            Literal["c", "python", "pyarrow", "python-fwf", "bigquery"]
        ] = None,
        encoding: Optional[str] = None,
        **kwargs,
    ) -> dataframe.DataFrame:
        table = bigquery.Table(self._create_session_table())

        if engine is not None and engine == "bigquery":
            if any(param is not None for param in (dtype, names)):
                not_supported = ("dtype", "names")
                raise NotImplementedError(
                    f"BigQuery engine does not support these arguments: {not_supported}"
                )

            if index_col is not None and (
                not index_col or not isinstance(index_col, str)
            ):
                raise NotImplementedError(
                    "BigQuery engine only supports a single column name for `index_col`."
                )

            # None value for index_col cannot be passed to read_gbq
            if index_col is None:
                index_col = ()

            # usecols should only be an iterable of strings (column names) for use as col_order in read_gbq.
            col_order: Tuple[Any, ...] = tuple()
            if usecols is not None:
                if isinstance(usecols, Iterable) and all(
                    isinstance(col, str) for col in usecols
                ):
                    col_order = tuple(col for col in usecols)
                else:
                    raise NotImplementedError(
                        "BigQuery engine only supports an iterable of strings for `usecols`."
                    )

            if not isinstance(filepath_or_buffer, str):
                raise NotImplementedError("BigQuery engine does not support buffers.")

            valid_encodings = {"UTF-8", "ISO-8859-1"}
            if encoding is not None and encoding not in valid_encodings:
                raise NotImplementedError(
                    f"BigQuery engine only supports the following encodings: {valid_encodings}"
                )

            job_config = bigquery.LoadJobConfig()
            job_config.create_disposition = bigquery.CreateDisposition.CREATE_IF_NEEDED
            job_config.source_format = bigquery.SourceFormat.CSV
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
            job_config.autodetect = True
            job_config.field_delimiter = sep
            job_config.encoding = encoding

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
            return self.read_gbq(
                f"SELECT * FROM `{table.table_id}`",
                index_col=index_col,
                col_order=col_order,
            )
        else:
            if any(arg in kwargs for arg in ("chunksize", "iterator")):
                raise NotImplementedError(
                    "'chunksize' and 'iterator' arguments are not supported."
                )

            self._check_file_size(filepath_or_buffer)
            pandas_df = pandas.read_csv(
                filepath_or_buffer,
                sep=sep,
                header=header,
                names=names,
                index_col=index_col,
                usecols=usecols,
                dtype=dtype,
                engine=engine,
                encoding=encoding,
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
        input_types: List[type],
        output_type: type,
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


def connect(context: Optional[bigquery_options.BigQueryOptions] = None) -> Session:
    return Session(context)
