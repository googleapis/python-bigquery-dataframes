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

from google.cloud import bigquery
import pytest

import bigframes

# https://cloud.google.com/bigquery/docs/locations
ALL_BIGQUERY_LOCATIONS = [
    "us-east5",
    "us-south1",
    "us-central1",
    "us-west4",
    "us-west2",
    "northamerica-northeast1",
    "us-east4",
    "us-west1",
    "us-west3",
    "southamerica-east1",
    "southamerica-west1",
    "us-east1",
    "northamerica-northeast2",
    "asia-south2",
    "asia-east2",
    "asia-southeast2",
    "australia-southeast2",
    "asia-south1",
    "asia-northeast2",
    "asia-northeast3",
    "asia-southeast1",
    "australia-southeast1",
    "asia-east1",
    "asia-northeast1",
    "europe-west1",
    "europe-west10",
    "europe-north1",
    "europe-west3",
    "europe-west2",
    "europe-southwest1",
    "europe-west8",
    "europe-west4",
    "europe-west9",
    "europe-west12",
    "europe-central2",
    "europe-west6",
    "me-central2",
    "me-central1",
    "me-west1",
    "me-central2",
    "me-central1",
    "me-west1",
    "africa-south1",
]

REP_ENABLED_BIGQUERY_LOCATIONS = [
    "me-central2",
    "europe-west9",
    "europe-west3",
    "us-east4",
    "us-west1",
]


def _assert_and_get_session(session: bigquery.Client):
    df = session.read_gbq(
        """
        SELECT "aaa" as name, 111 as number
        UNION ALL
        SELECT "bbb" as name, 222 as number
        UNION ALL
        SELECT "aaa" as name, 333 as number
    """
    )

    assert df.query_job.location == session.bqclient.location

    result = (
        df[["name", "number"]]
        .groupby("name")
        .sum(numeric_only=True)
        .sort_values("number", ascending=False)
        .head()
    )

    assert result.query_job.location == session.bqclient.location


@pytest.mark.parametrize("bigquery_location", ALL_BIGQUERY_LOCATIONS)
def test_bq_location(bigquery_location):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(location=bigquery_location)
    )

    assert session.bqclient.location == bigquery_location

    # by default global endpoint is used
    assert (
        session.bqclient._connection.API_BASE_URL == "https://bigquery.googleapis.com"
    )

    # assert that bigframes session honors the location
    _assert_and_get_session(session)


@pytest.mark.parametrize(
    "bigquery_location",
    REP_ENABLED_BIGQUERY_LOCATIONS,
)
def test_bq_rep_endpoints(bigquery_location):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(
            location=bigquery_location, use_regional_endpoints=True
        )
    )

    assert session.bqclient.location == bigquery_location
    assert (
        session.bqclient._connection.API_BASE_URL
        == "https://bigquery.{location}.rep.googleapis.com".format(
            location=bigquery_location
        )
    )

    # assert that bigframes session honors the location
    _assert_and_get_session(session)


@pytest.mark.parametrize(
    "bigquery_location",
    set(ALL_BIGQUERY_LOCATIONS) - set(REP_ENABLED_BIGQUERY_LOCATIONS),
)
def test_bq_lep_endpoints(bigquery_location):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(
            location=bigquery_location, use_regional_endpoints=True
        )
    )

    assert session.bqclient.location == bigquery_location
    assert (
        session.bqclient._connection.API_BASE_URL
        == "https://{location}-bigquery.googleapis.com".format(
            location=bigquery_location
        )
    )

    # We are not testing BQ query execution with LEP endpoints
    # as that requires the project to be allowlisted for LEP access.
    # We could hardcode one project which is allowlisted but then not every
    # open source will have access to that.
