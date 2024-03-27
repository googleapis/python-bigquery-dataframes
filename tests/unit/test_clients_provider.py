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

import pytest

import bigframes
import bigframes.session.clients
from tests import config


def test_bq_client_default():
    clients_provider = bigframes.session.clients.ClientsProvider()

    assert (
        clients_provider.bqclient.location is None
    )  # would imply default location "US" for the client
    assert (
        clients_provider.bqclient._connection.API_BASE_URL
        == "https://bigquery.googleapis.com"
    )


@pytest.mark.parametrize(
    "bigquery_location",
    config.ALL_BIGQUERY_LOCATIONS,
)
def test_bq_client_default_endpoint(bigquery_location):
    clients_provider = bigframes.session.clients.ClientsProvider(
        location=bigquery_location
    )

    assert clients_provider.bqclient.location == bigquery_location
    assert (
        clients_provider.bqclient._connection.API_BASE_URL
        == "https://bigquery.googleapis.com"
    )


@pytest.mark.parametrize(
    "bigquery_location",
    config.REP_ENABLED_BIGQUERY_LOCATIONS,
)
def test_bq_client_rep_endpoint(bigquery_location):
    clients_provider = bigframes.session.clients.ClientsProvider(
        location=bigquery_location, use_regional_endpoints=True
    )

    assert clients_provider.bqclient.location == bigquery_location
    assert (
        clients_provider.bqclient._connection.API_BASE_URL
        == "https://bigquery.{location}.rep.googleapis.com".format(
            location=bigquery_location
        )
    )


@pytest.mark.parametrize(
    "bigquery_location",
    config.LEP_ENABLED_BIGQUERY_LOCATIONS,
)
def test_bq_client_lep_endpoint(bigquery_location):
    clients_provider = bigframes.session.clients.ClientsProvider(
        location=bigquery_location, use_regional_endpoints=True
    )

    assert clients_provider.bqclient.location == bigquery_location
    assert (
        clients_provider.bqclient._connection.API_BASE_URL
        == "https://{location}-bigquery.googleapis.com".format(
            location=bigquery_location
        )
    )
