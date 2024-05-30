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

import ibis
import ibis.expr.schema

import bigframes.core.schema as bf_schema
import bigframes.dtypes


def convert_bf_schema(schema: bf_schema.ArraySchema) -> ibis.expr.schema.Schema:
    """
    Convert bigframes schema to ibis schema. This is unambigous as every bigframes type is backed by a specific SQL/ibis dtype.
    """
    names = schema.names
    types = [
        bigframes.dtypes.bigframes_dtype_to_ibis_dtype(bf_type)
        for bf_type in schema.dtypes
    ]
    return ibis.schema(names=names, types=types)
