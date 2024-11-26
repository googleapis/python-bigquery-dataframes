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

"""This module integrates BigQuery built-in functions for use with DataFrame objects,
such as array functions:
https://cloud.google.com/bigquery/docs/reference/standard-sql/array_functions. """

from bigframes.bigquery._operations.approx_agg import approx_top_count
from bigframes.bigquery._operations.array import (
    array_agg,
    array_length,
    array_to_string,
)
from bigframes.bigquery._operations.json import (
    json_extract,
    json_extract_array,
    json_extract_string_array,
    json_set,
)
from bigframes.bigquery._operations.search import create_vector_index, vector_search
from bigframes.bigquery._operations.struct import struct

__all__ = [
    "array_length",
    "array_agg",
    "array_to_string",
    "json_set",
    "json_extract",
    "json_extract_array",
    "json_extract_string_array",
    "approx_top_count",
    "struct",
    "create_vector_index",
    "vector_search",
]
