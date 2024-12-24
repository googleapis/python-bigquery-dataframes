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

"""BigQuery DataFrames provides a DataFrame API scaled by the BigQuery engine."""

import IPython

from bigframes._config import option_context, options
from bigframes._config.bigquery_options import BigQueryOptions
from bigframes.core.global_session import close_session, get_global_session
import bigframes.enums as enums
import bigframes.exceptions as exceptions
from bigframes.session import connect, Session
from bigframes.version import __version__


def load_ipython_extension(ipython):
    """Called by IPython when this module is loaded as an IPython extension."""
    # Import here to avoid circular imports.
    import bigframes.magics

    if ipython is not None:
        bigframes.magics.load_ipython_extension(ipython)


# Automatically load the extension, to make the magics easier to discover.
load_ipython_extension(IPython.get_ipython())


__all__ = [
    "options",
    "BigQueryOptions",
    "get_global_session",
    "close_session",
    "enums",
    "exceptions",
    "connect",
    "Session",
    "__version__",
    "option_context",
]
