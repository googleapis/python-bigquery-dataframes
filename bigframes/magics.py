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

import bigquery_magics  # type: ignore
import bigquery_magics.config  # type: ignore


def load_ipython_extension(ipython):
    """Called by IPython when this module is loaded as an IPython extension."""
    bigquery_magics.load_ipython_extension(ipython)

    if bigquery_magics.context.credentials is not None:
        # The %%bigquery magics must have been run before BigQuery DataFrames
        # was imported. In this case, we don't want to break any existing
        # notebooks, so don't make any BigQuery DataFrames changes to the
        # magics.
        return

    bigquery_magics.context = Context()


class Context(bigquery_magics.config.Context):
    """A provider for bigquery-magics configuration, derived from bigframes
    global options and global default session.
    """

    def __init__(self):
        pass
