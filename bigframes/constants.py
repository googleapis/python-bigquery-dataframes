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

"""Constants used across BigQuery DataFrames.

This module should not depend on any others in the package.
"""

import datetime

DEFAULT_EXPIRATION = datetime.timedelta(days=7)

# https://cloud.google.com/bigquery/docs/locations
BIGQUERY_REGIONS = frozenset(
    {
        "africa-south1",
        "asia-east1",
        "asia-east2",
        "asia-northeast1",
        "asia-northeast2",
        "asia-northeast3",
        "asia-south1",
        "asia-south2",
        "asia-southeast1",
        "asia-southeast2",
        "australia-southeast1",
        "australia-southeast2",
        "europe-central2",
        "europe-north1",
        "europe-southwest1",
        "europe-west1",
        "europe-west10",
        "europe-west12",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west8",
        "europe-west9",
        "me-central1",
        "me-central2",
        "me-west1",
        "northamerica-northeast1",
        "northamerica-northeast2",
        "southamerica-east1",
        "southamerica-west1",
        "us-central1",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-south1",
        "us-west1",
        "us-west2",
        "us-west3",
        "us-west4",
    }
)
BIGQUERY_MULTIREGIONS = frozenset(
    {
        "EU",
        "US",
    }
)
ALL_BIGQUERY_LOCATIONS = frozenset(BIGQUERY_REGIONS.union(BIGQUERY_MULTIREGIONS))

# https://cloud.google.com/storage/docs/regional-endpoints
REP_ENABLED_BIGQUERY_LOCATIONS = frozenset(
    {
        "europe-west3",
        "europe-west8",
        "europe-west9",
        "me-central2",
        "us-central1",
        "us-central2",
        "us-east1",
        "us-east4",
        "us-east5",
        "us-east7",
        "us-south1",
        "us-west1",
        "us-west2",
        "us-west3",
        "us-west4",
    }
)

# https://cloud.google.com/storage/docs/locational-endpoints
LEP_ENABLED_BIGQUERY_LOCATIONS = frozenset(
    ALL_BIGQUERY_LOCATIONS - REP_ENABLED_BIGQUERY_LOCATIONS
)

# BigQuery default is 10000, leave 100 for overhead
MAX_COLUMNS = 9900

SUGGEST_PEEK_PREVIEW = "Use .peek(n) to preview n arbitrary rows."
