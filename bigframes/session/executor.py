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

from __future__ import annotations

import abc
import dataclasses
import datetime
import functools
import itertools
from typing import Any, Iterator, Literal, Optional, Sequence, Union

from google.cloud import bigquery, bigquery_storage_v1
import pandas as pd
import pyarrow
import pyarrow as pa

import bigframes
import bigframes.core
from bigframes.core import pyarrow_utils
import bigframes.core.schema
import bigframes.session._io.pandas as io_pandas
import bigframes.session.execution_spec as ex_spec

_ROW_LIMIT_EXCEEDED_TEMPLATE = (
    "Execution has downloaded {result_rows} rows so far, which exceeds the "
    "limit of {maximum_result_rows}. You can adjust this limit by setting "
    "`bpd.options.compute.maximum_result_rows`."
)


class ExecuteResult(abc.ABC):
    @property
    @abc.abstractmethod
    def query_job(self) -> Optional[bigquery.QueryJob]:
        ...

    @property
    @abc.abstractmethod
    def total_bytes(self) -> Optional[int]:
        ...

    @property
    @abc.abstractmethod
    def total_rows(self) -> Optional[int]:
        ...

    @property
    @abc.abstractmethod
    def total_bytes_processed(self) -> Optional[int]:
        ...

    @property
    @abc.abstractmethod
    def schema(self) -> bigframes.core.schema.ArraySchema:
        ...

    @abc.abstractmethod
    def _get_arrow_batches(self) -> Iterator[pyarrow.RecordBatch]:
        ...

    @property
    def arrow_batches(self) -> Iterator[pyarrow.RecordBatch]:
        result_rows = 0

        for batch in self._get_arrow_batches():
            batch = pyarrow_utils.cast_batch(batch, self.schema.to_pyarrow())
            result_rows += batch.num_rows

            maximum_result_rows = bigframes.options.compute.maximum_result_rows
            if maximum_result_rows is not None and result_rows > maximum_result_rows:
                message = bigframes.exceptions.format_message(
                    _ROW_LIMIT_EXCEEDED_TEMPLATE.format(
                        result_rows=result_rows,
                        maximum_result_rows=maximum_result_rows,
                    )
                )
                raise bigframes.exceptions.MaximumResultRowsExceeded(message)

            yield batch

    def to_arrow_table(self) -> pyarrow.Table:
        # Need to provide schema if no result rows, as arrow can't infer
        # If ther are rows, it is safest to infer schema from batches.
        # Any discrepencies between predicted schema and actual schema will produce errors.
        batches = iter(self.arrow_batches)
        peek_it = itertools.islice(batches, 0, 1)
        peek_value = list(peek_it)
        # TODO: Enforce our internal schema on the table for consistency
        if len(peek_value) > 0:
            return pyarrow.Table.from_batches(
                itertools.chain(peek_value, batches),  # reconstruct
            )
        else:
            return self.schema.to_pyarrow().empty_table()

    def to_pandas(self) -> pd.DataFrame:
        return io_pandas.arrow_to_pandas(self.to_arrow_table(), self.schema)

    def to_pandas_batches(
        self, page_size: Optional[int] = None, max_results: Optional[int] = None
    ) -> Iterator[pd.DataFrame]:
        assert (page_size is None) or (page_size > 0)
        assert (max_results is None) or (max_results > 0)
        batch_iter: Iterator[
            Union[pyarrow.Table, pyarrow.RecordBatch]
        ] = self.arrow_batches
        if max_results is not None:
            batch_iter = pyarrow_utils.truncate_pyarrow_iterable(
                batch_iter, max_results
            )

        if page_size is not None:
            batches_iter = pyarrow_utils.chunk_by_row_count(batch_iter, page_size)
            batch_iter = map(
                lambda batches: pyarrow.Table.from_batches(batches), batches_iter
            )

        yield from map(
            functools.partial(io_pandas.arrow_to_pandas, schema=self.schema),
            batch_iter,
        )

    def to_py_scalar(self):
        columns = list(self.to_arrow_table().to_pydict().values())
        if len(columns) != 1:
            raise ValueError(
                f"Expected single column result, got {len(columns)} columns."
            )
        column = columns[0]
        if len(column) != 1:
            raise ValueError(f"Expected single row result, got {len(column)} rows.")
        return column[0]


class LocalExecuteResult(ExecuteResult):
    def __init__(self, data: pa.Table, bf_schema: bigframes.core.schema.ArraySchema):
        self._data = data
        self._schema = bf_schema

    @property
    def query_job(self) -> Optional[bigquery.QueryJob]:
        return None

    @property
    def total_bytes(self) -> Optional[int]:
        return None

    @property
    def total_rows(self) -> Optional[int]:
        return self._data.num_rows

    @property
    def total_bytes_processed(self) -> Optional[int]:
        return None

    @property
    def schema(self) -> bigframes.core.schema.ArraySchema:
        return self._schema

    def _get_arrow_batches(self) -> Iterator[pyarrow.RecordBatch]:
        return iter(self._data.to_batches())


class EmptyExecuteResult(ExecuteResult):
    def __init__(
        self,
        bf_schema: bigframes.core.schema.ArraySchema,
        query_job: Optional[bigquery.QueryJob] = None,
    ):
        self._schema = bf_schema
        self._query_job = query_job

    @property
    def query_job(self) -> Optional[bigquery.QueryJob]:
        return self._query_job

    @property
    def total_bytes(self) -> Optional[int]:
        return None

    @property
    def total_rows(self) -> Optional[int]:
        return 0

    @property
    def total_bytes_processed(self) -> Optional[int]:
        if self.query_job:
            return self.query_job.total_bytes_processed
        return None

    @property
    def schema(self) -> bigframes.core.schema.ArraySchema:
        return self._schema

    def _get_arrow_batches(self) -> Iterator[pyarrow.RecordBatch]:
        return iter([])


class BQTableExecuteResult(ExecuteResult):
    def __init__(
        self,
        data: bigquery.TableReference,
        bf_schema: bigframes.core.schema.ArraySchema,
        bq_client: bigquery.Client,
        storage_client: bigquery_storage_v1.BigQueryReadClient,
        *,
        query_job: Optional[bigquery.QueryJob] = None,
        snapshot_time: Optional[datetime.datetime] = None,
        limit: Optional[int] = None,
        selected_fields: Optional[Sequence[str]] = None,
        sql_predicate: Optional[str] = None,
    ):
        self._data = data
        self._schema = bf_schema
        self._query_job = query_job
        self._bqclient = bq_client
        self._storage_client = storage_client
        self._snapshot_time = snapshot_time
        self._limit = limit
        self._selected_fields = selected_fields
        self._predicate = sql_predicate

    @property
    def query_job(self) -> Optional[bigquery.QueryJob]:
        return self._query_job

    @property
    def total_bytes(self) -> Optional[int]:
        return None

    @property
    def total_rows(self) -> Optional[int]:
        return self._get_table_metadata(self._data).num_rows

    @functools.cache
    def _get_table_metadata(self) -> bigquery.Table:
        return self._bqclient.get_table(self._data)

    @property
    def total_bytes_processed(self) -> Optional[int]:
        if self.query_job:
            return self.query_job.total_bytes_processed
        return None

    @property
    def schema(self) -> bigframes.core.schema.ArraySchema:
        return self._schema

    def _get_arrow_batches(self) -> Iterator[pyarrow.RecordBatch]:
        import google.cloud.bigquery_storage_v1.types as bq_storage_types
        from google.protobuf import timestamp_pb2

        table_mod_options = {}
        read_options_dict: dict[str, Any] = {}
        if self._selected_fields:
            read_options_dict["selected_fields"] = list(self._selected_fields)
        if self._predicate:
            read_options_dict["row_restriction"] = self._predicate
        read_options = bq_storage_types.ReadSession.TableReadOptions(
            **read_options_dict
        )

        if self._snapshot_time:
            snapshot_time = timestamp_pb2.Timestamp()
            snapshot_time.FromDatetime(self._snapshot_time)
            table_mod_options["snapshot_time"] = snapshot_time = snapshot_time
        table_mods = bq_storage_types.ReadSession.TableModifiers(**table_mod_options)

        requested_session = bq_storage_types.stream.ReadSession(
            table=self._data.to_bqstorage(),
            data_format=bq_storage_types.DataFormat.ARROW,
            read_options=read_options,
            table_modifiers=table_mods,
        )
        # Single stream to maintain ordering
        request = bq_storage_types.CreateReadSessionRequest(
            parent=f"projects/{self._data.project}",
            read_session=requested_session,
            max_stream_count=1,
        )
        session = self._storage_client.create_read_session(request=request)

        if not session.streams:
            batches: Iterator[pa.RecordBatch] = iter([])
        else:
            reader = self._storage_client.read_rows(session.streams[0].name)
            rowstream = reader.rows()

            def process_page(page):
                pa_batch = page.to_arrow()
                return pa.RecordBatch.from_arrays(
                    pa_batch.columns, names=self.schema.names
                )

            batches = map(process_page, rowstream.pages)

        return batches


@dataclasses.dataclass(frozen=True)
class HierarchicalKey:
    columns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CacheConfig(abc.ABC):
    optimize_for: Union[Literal["auto", "head"], HierarchicalKey] = "auto"
    if_cached: Literal["reuse-strict", "reuse-any", "replace"] = "reuse-any"


class Executor(abc.ABC):
    """
    Interface for an executor, which compiles and executes ArrayValue objects.
    """

    def to_sql(
        self,
        array_value: bigframes.core.ArrayValue,
        offset_column: Optional[str] = None,
        ordered: bool = False,
        enable_cache: bool = True,
    ) -> str:
        """
        Convert an ArrayValue to a sql query that will yield its value.
        """
        raise NotImplementedError("to_sql not implemented for this executor")

    @abc.abstractmethod
    def execute(
        self,
        array_value: bigframes.core.ArrayValue,
        execution_spec: ex_spec.ExecutionSpec,
    ) -> ExecuteResult:
        """
        Execute the ArrayValue.
        """
        ...

    def dry_run(
        self, array_value: bigframes.core.ArrayValue, ordered: bool = True
    ) -> bigquery.QueryJob:
        """
        Dry run executing the ArrayValue.

        Does not actually execute the data but will get stats and indicate any invalid query errors.
        """
        raise NotImplementedError("dry_run not implemented for this executor")

    def cached(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        config: CacheConfig,
    ) -> None:
        raise NotImplementedError("cached not implemented for this executor")
