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

from typing import Union

import google.cloud.bigquery.table

import bigframes
import bigframes.exceptions
from bigframes.session import executor

ROW_LIMIT_EXCEEDED_TEMPLATE = (
    "Execution would result in {total_rows} rows, which "
    "exceeds the limit of {maximum_rows_downloaded}. "
    "You can adjust this limit by setting "
    "`bpd.options.compute.maximum_rows_downloaded`."
)


def check_row_limit(
    *,
    result: Union[google.cloud.bigquery.table.RowIterator, executor.ExecuteResult],
) -> None:
    """
    Checks if the total number of rows exceeds the configured limit.

    Args:
        result: Result of a query.

    Raises:
        An instance of `exception_type` if the limit is exceeded.
    """
    total_rows = result.total_rows
    maximum_rows_downloaded = bigframes.options.compute.maximum_rows_downloaded

    if total_rows is None or maximum_rows_downloaded is None:
        return

    if total_rows > maximum_rows_downloaded:
        message = bigframes.exceptions.format_message(
            ROW_LIMIT_EXCEEDED_TEMPLATE.format(
                total_rows=total_rows,
                maximum_rows_downloaded=maximum_rows_downloaded,
            )
        )
        raise bigframes.exceptions.MaximumRowsDownloadedExceeded(message)
