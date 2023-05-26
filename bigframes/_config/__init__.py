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

"""
Configuration for bigframes. Do not depend on other parts of BigFrames from
this package.
"""

import bigframes._config.bigquery_options as bigquery_options


class Options:
    """Global options affecting BigFrames behavior."""

    def __init__(self):
        self._bigquery_options = bigquery_options.BigQueryOptions()

    @property
    def bigquery(self) -> bigquery_options.BigQueryOptions:
        """Options to use with the BigQuery engine."""
        return self._bigquery_options


options = Options()
"""Global options for default session."""


__all__ = (
    "Options",
    "options",
)
