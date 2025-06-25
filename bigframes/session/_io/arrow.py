# Copyright 2025 Google LLC
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

"""Private helpers for reading pyarrow objects."""

from __future__ import annotations

import pyarrow as pa

from bigframes import dataframe
import bigframes.core as core
import bigframes.dtypes
from bigframes.core import local_data, pyarrow_utils
import bigframes.core.blocks as blocks
import bigframes.core.guid
import bigframes.core.schema as schemata
import bigframes.session


def create_dataframe_from_arrow_table(
    pa_table: pa.Table, *, session: bigframes.session.Session
) -> dataframe.DataFrame:
    """Convert a PyArrow Table into a BigQuery DataFrames DataFrame.

    This DataFrame will wrap a LocalNode, meaning the data is processed locally.

    Args:
        pa_table (pyarrow.Table):
            The PyArrow Table to convert.
        session (bigframes.session.Session):
            The BigQuery DataFrames session to associate with the new DataFrame.

    Returns:
        bigframes.dataframe.DataFrame:
            A new DataFrame representing the data from the PyArrow table.
    """
    # TODO(tswast): Use array_value.promote_offsets() instead once that node is
    # supported by the local engine.
    offsets_col = bigframes.core.guid.generate_guid()
    # TODO(https://github.com/googleapis/python-bigquery-dataframes/issues/859):
    # Allow users to specify the "total ordering" column(s) or allow multiple
    # such columns.
    pa_table = pyarrow_utils.append_offsets(pa_table, offsets_col=offsets_col)

    # We use the ManagedArrowTable constructor directly, because the
    # results of to_arrow() should be the source of truth with regards
    # to canonical formats since it comes from either the BQ Storage
    # Read API or has been transformed by google-cloud-bigquery to look
    # like the output of the BQ Storage Read API.
    schema_items = []
    for field in pa_table.schema:
        bf_dtype = bigframes.dtypes.arrow_dtype_to_bigframes_dtype(field.type, allow_lossless_cast=True)
        schema_items.append(schemata.SchemaItem(field.name, bf_dtype))
    bf_schema = schemata.ArraySchema(tuple(schema_items))

    mat = local_data.ManagedArrowTable(
        pa_table,
        bf_schema,
    )
    mat.validate()

    array_value = core.ArrayValue.from_managed(mat, session)
    block = blocks.Block(
        array_value,
        (offsets_col,),
        [field.name for field in pa_table.schema if field.name != offsets_col],
        (None,),
    )
    return dataframe.DataFrame(block)
