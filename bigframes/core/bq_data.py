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

from __future__ import annotations

import dataclasses
import datetime
import functools
import typing
from typing import Optional, Sequence, Tuple

import google.cloud.bigquery as bq

import bigframes.core.schema

if typing.TYPE_CHECKING:
    import bigframes.core.ordering as orderings


@dataclasses.dataclass(frozen=True)
class GbqTable:
    project_id: str = dataclasses.field()
    dataset_id: str = dataclasses.field()
    table_id: str = dataclasses.field()
    physical_schema: Tuple[bq.SchemaField, ...] = dataclasses.field()
    is_physically_stored: bool = dataclasses.field()
    cluster_cols: typing.Optional[Tuple[str, ...]]

    @staticmethod
    def from_table(table: bq.Table, columns: Sequence[str] = ()) -> GbqTable:
        # Subsetting fields with columns can reduce cost of row-hash default ordering
        if columns:
            schema = tuple(item for item in table.schema if item.name in columns)
        else:
            schema = tuple(table.schema)
        return GbqTable(
            project_id=table.project,
            dataset_id=table.dataset_id,
            table_id=table.table_id,
            physical_schema=schema,
            is_physically_stored=(table.table_type in ["TABLE", "MATERIALIZED_VIEW"]),
            cluster_cols=None
            if table.clustering_fields is None
            else tuple(table.clustering_fields),
        )

    def get_table_ref(self) -> bq.TableReference:
        return bq.TableReference(
            bq.DatasetReference(self.project_id, self.dataset_id), self.table_id
        )

    @property
    @functools.cache
    def schema_by_id(self):
        return {col.name: col for col in self.physical_schema}


@dataclasses.dataclass(frozen=True)
class BigqueryDataSource:
    """
    Google BigQuery Data source.

    This should not be modified once defined, as all attributes contribute to the default ordering.
    """

    table: GbqTable
    schema: bigframes.core.schema.ArraySchema
    at_time: typing.Optional[datetime.datetime] = None
    # Added for backwards compatibility, not validated
    sql_predicate: typing.Optional[str] = None
    ordering: typing.Optional[orderings.RowOrdering] = None
    # Optimization field
    n_rows: Optional[int] = None
