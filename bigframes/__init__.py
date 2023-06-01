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

"""Bigframes provides a DataFrame API for BigQuery."""

from bigframes._config import options
from bigframes._config.bigquery_options import BigQueryOptions
from bigframes.bigframes import concat
from bigframes.dataframe import DataFrame
from bigframes.remote_function import remote_function
from bigframes.series import Series
from bigframes.session import connect, Session
from bigframes.version import __version__

__all__ = [
    "BigQueryOptions",
    "concat",
    "connect",
    "Session",
    "DataFrame",
    "Series",
    "options",
    "remote_function",
    "__version__",
]
