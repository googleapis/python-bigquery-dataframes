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

from typing import Optional

def check_row_limit(
    total_rows: int,
    maximum_rows_downloaded: Optional[int],
    exception_type: type,
    operation_name: str = "Query",
) -> None:
    """
    Checks if the total number of rows exceeds the configured limit.

    Args:
        total_rows: The total number of rows to be processed or downloaded.
        maximum_rows_downloaded: The configured maximum number of rows allowed.
            If None, no check is performed.
        exception_type: The type of exception to raise if the limit is exceeded.
        operation_name: A descriptive name of the operation being performed
            (e.g., "Download", "Conversion to pandas"). Used in the
            exception message.

    Raises:
        An instance of `exception_type` if the limit is exceeded.
    """
    if maximum_rows_downloaded is not None and total_rows > maximum_rows_downloaded:
        message = (
            f"{operation_name} would result in {total_rows} rows, which "
            f"exceeds the limit of {maximum_rows_downloaded}. "
            "You can adjust this limit by setting "
            "`bigframes.options.compute.maximum_rows_downloaded`."
        )
        raise exception_type(message)
