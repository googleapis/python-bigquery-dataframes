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

import typing

from google.cloud import bigquery
from google.cloud.bigquery_storage import types as bqstorage_types
import pandas
import pandas.testing
import pytest

import bigframes
import bigframes.constants
import bigframes.session.clients


def _assert_bq_execution_location(
    session: bigframes.Session, expected_location: typing.Optional[str] = None
):
    df = session.read_gbq(
        """
        SELECT "aaa" as name, 111 as number
        UNION ALL
        SELECT "bbb" as name, 222 as number
        UNION ALL
        SELECT "aaa" as name, 333 as number
    """
    )

    if expected_location is None:
        expected_location = session._location

    assert typing.cast(bigquery.QueryJob, df.query_job).location == expected_location

    # Ensure operation involving BQ client suceeds
    result = (
        df[["name", "number"]]
        .groupby("name")
        .sum(numeric_only=True)
        .sort_values("number", ascending=False)
        .head()
    )

    assert (
        typing.cast(bigquery.QueryJob, result.query_job).location == expected_location
    )

    expected_result = pandas.DataFrame(
        {"number": [444, 222]}, index=pandas.Index(["aaa", "bbb"], name="name")
    )
    pandas.testing.assert_frame_equal(
        expected_result, result.to_pandas(), check_dtype=False, check_index_type=False
    )

    # Ensure BQ Storage Read client operation succceeds
    table = result.query_job.destination
    requested_session = bqstorage_types.ReadSession(  # type: ignore[attr-defined]
        table=f"projects/{table.project}/datasets/{table.dataset_id}/tables/{table.table_id}",
        data_format=bqstorage_types.DataFormat.ARROW,  # type: ignore[attr-defined]
    )
    read_session = session.bqstoragereadclient.create_read_session(
        parent=f"projects/{table.project}",
        read_session=requested_session,
        max_stream_count=1,
    )
    reader = session.bqstoragereadclient.read_rows(read_session.streams[0].name)
    frames = []
    for message in reader.rows().pages:
        frames.append(message.to_dataframe())
    read_dataframe = pandas.concat(frames)
    # normalize before comparing since we lost some of the bigframes column
    # naming abtractions in the direct read of the destination table
    read_dataframe = read_dataframe.set_index("name")
    read_dataframe.columns = result.columns
    pandas.testing.assert_frame_equal(expected_result, read_dataframe)


def test_bq_location_default():
    session = bigframes.Session()

    assert session.bqclient.location == "US"

    # by default global endpoint is used
    assert (
        session.bqclient._connection.API_BASE_URL == "https://bigquery.googleapis.com"
    )

    # assert that bigframes session honors the location
    _assert_bq_execution_location(session)


@pytest.mark.parametrize(
    "bigquery_location",
    # Sort the set to avoid nondeterminism.
    sorted(bigframes.constants.ALL_BIGQUERY_LOCATIONS),
)
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
    _assert_bq_execution_location(session)


@pytest.mark.parametrize(
    ("set_location", "resolved_location"),
    # Sort the set to avoid nondeterminism.
    [
        (loc.capitalize(), loc)
        for loc in sorted(bigframes.constants.ALL_BIGQUERY_LOCATIONS)
    ],
)
def test_bq_location_non_canonical(set_location, resolved_location):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(location=set_location)
    )

    assert session.bqclient.location == resolved_location

    # by default global endpoint is used
    assert (
        session.bqclient._connection.API_BASE_URL == "https://bigquery.googleapis.com"
    )

    # assert that bigframes session honors the location
    _assert_bq_execution_location(session, resolved_location)


@pytest.mark.parametrize(
    "bigquery_location",
    # Sort the set to avoid nondeterminism.
    sorted(bigframes.constants.REP_ENABLED_BIGQUERY_LOCATIONS),
)
def test_bq_rep_endpoints(bigquery_location):
    session = bigframes.Session(
        context=bigframes.BigQueryOptions(
            location=bigquery_location, use_regional_endpoints=True
        )
    )

    # Verify that location and endpoint is correctly set for the BigQuery API
    # client
    assert session.bqclient.location == bigquery_location
    assert (
        session.bqclient._connection.API_BASE_URL
        == "https://bigquery.{location}.rep.googleapis.com".format(
            location=bigquery_location
        )
    )

    # Verify that endpoint is correctly set for the BigQuery Storage API client
    # TODO(shobs): Figure out if we can verify that location is set in the
    # BigQuery Storage API client.
    assert (
        session.bqstoragereadclient.api_endpoint
        == f"bigquerystorage.{bigquery_location}.rep.googleapis.com"
    )

    # assert that bigframes session honors the location
    _assert_bq_execution_location(session)


def test_clients_provider_no_location():
    with pytest.raises(ValueError, match="Must set location to use regional endpoints"):
        bigframes.session.clients.ClientsProvider(use_regional_endpoints=True)


@pytest.mark.parametrize(
    "bigquery_location",
    # Sort the set to avoid nondeterminism.
    sorted(bigframes.constants.REP_NOT_ENABLED_BIGQUERY_LOCATIONS),
)
def test_clients_provider_use_regional_endpoints_non_rep_locations(bigquery_location):
    with pytest.raises(
        ValueError,
        match=f"not .*available in the location {bigquery_location}",
    ):
        bigframes.session.clients.ClientsProvider(
            location=bigquery_location, use_regional_endpoints=True
        )


@pytest.mark.parametrize(
    "bigquery_location",
    # Sort the set to avoid nondeterminism.
    sorted(bigframes.constants.REP_NOT_ENABLED_BIGQUERY_LOCATIONS),
)
def test_session_init_fails_to_use_regional_endpoints_non_rep_endpoints(
    bigquery_location,
):
    with pytest.raises(
        ValueError,
        match=f"not .*available in the location {bigquery_location}",
    ):
        bigframes.Session(
            context=bigframes.BigQueryOptions(
                location=bigquery_location, use_regional_endpoints=True
            )
        )
