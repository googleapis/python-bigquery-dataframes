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

import dataclasses
from typing import Mapping, Optional, Union
import weakref

import polars

import bigframes
import bigframes.clients
import bigframes.core.blocks
import bigframes.core.compile.polars
import bigframes.core.ordering
import bigframes.dataframe
import bigframes.session.clients
import bigframes.session.executor
import bigframes.session.metrics


# Does not support to_sql, export_gbq, export_gcs, dry_run, peek, head, get_row_count, cached
@dataclasses.dataclass
class TestExecutor(bigframes.session.executor.Executor):
    compiler = bigframes.core.compile.polars.PolarsCompiler()

    def execute(
        self,
        array_value: bigframes.core.ArrayValue,
        *,
        ordered: bool = True,
        col_id_overrides: Mapping[str, str] = {},
        use_explicit_destination: bool = False,
        get_size_bytes: bool = False,
        page_size: Optional[int] = None,
        max_results: Optional[int] = None,
    ):
        """
        Execute the ArrayValue, storing the result to a temporary session-owned table.
        """
        lazy_frame: polars.LazyFrame = self.compiler.compile(array_value)
        pa_table = lazy_frame.collect().to_arrow()
        # Currently, pyarrow types might not quite be exactly the ones in the bigframes schema.
        # Nullability may be different, and might use large versions of list, string datatypes.
        return bigframes.session.executor.ExecuteResult(
            arrow_batches=lambda: pa_table.to_batches(),
            schema=array_value.schema,
            total_bytes=pa_table.nbytes,
            total_rows=pa_table.num_rows,
        )


class TestSession(bigframes.session.Session):
    def __init__(self):
        self._location = None  # type: ignore
        self._bq_kms_key_name = None  # type: ignore
        self._clients_provider = None  # type: ignore
        self.ibis_client = None  # type: ignore
        self._bq_connection = None  # type: ignore
        self._skip_bq_connection_check = True
        self._session_id: str = "test_session"
        self._objects: list[
            weakref.ReferenceType[
                Union[
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
        self._temp_storage_manager = None  # type: ignore
        self._executor = TestExecutor()
        self._loader = None  # type: ignore

    def read_pandas(self, pandas_dataframe):
        # override read_pandas to always keep data local-only
        local_block = bigframes.core.blocks.Block.from_local(pandas_dataframe, self)
        return bigframes.dataframe.DataFrame(local_block)
