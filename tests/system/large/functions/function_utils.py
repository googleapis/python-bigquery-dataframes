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

from tests.system.utils import delete_cloud_function


def cleanup_function_assets(
    bigframes_func,
    bigquery_client,
    cloudfunctions_client=None,
    ignore_failures=True,
) -> None:
    """Clean up the GCP assets behind a bigframess function."""

    # Clean up bigframes function.
    try:
        bigquery_client.delete_routine(
            bigframes_func.bigframes_bigquery_function
        )
    except Exception:
        # By default don't raise exception in cleanup.
        if not ignore_failures:
            raise

    # Clean up cloud function
    try:
        delete_cloud_function(
            cloudfunctions_client, bigframes_func.bigframes_cloud_function
        )
    except Exception:
        # By default don't raise exception in cleanup.
        if not ignore_failures:
            raise
