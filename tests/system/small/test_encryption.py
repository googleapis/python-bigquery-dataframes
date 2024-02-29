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

import pandas
import pytest

import bigframes


@pytest.fixture(scope="module")
def bq_cmek() -> str:
    """Customer managed encryption key to encrypt BigQuery data at rest.

    This is of the form projects/PROJECT_ID/locations/LOCATION/keyRings/KEY_RING/cryptoKeys/KEY

    See https://cloud.google.com/bigquery/docs/customer-managed-encryption for steps.
    """

    # We are keeping this empty, and the dependent tests are skipped if it is empty.
    # In local testing we can set a key here and run the tests.
    # TODO(shobs): Automate the tests depending on this fixture by either creating
    # a static key in the test project or automating the key creation during the test.
    return ""


@pytest.fixture(scope="module")
def session_with_bq_cmek(bq_cmek) -> bigframes.Session:
    session = bigframes.Session(bigframes.BigQueryOptions(kms_key_name=bq_cmek))

    return session


def _assert_bq_table_is_encrypted(
    df: bigframes.dataframe.DataFrame,
    cmek: str,
    session: bigframes.Session,
):
    # Materialize the data in BQ
    repr(df)

    # The df should be backed by a query job with intended encryption on the result table
    assert df.query_job is not None
    assert df.query_job.destination_encryption_configuration.kms_key_name.startswith(
        cmek
    )

    # The result table should exist with the intended encryption
    table = session.bqclient.get_table(df.query_job.destination)
    assert table.encryption_configuration.kms_key_name == cmek


def test_read_gbq(bq_cmek, session_with_bq_cmek, scalars_table_id):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Session should have cmek set in the default query job config
    assert (
        session_with_bq_cmek.bqclient.default_query_job_config.destination_encryption_configuration.kms_key_name
        == bq_cmek
    )

    # Read the BQ table
    df = session_with_bq_cmek.read_gbq(scalars_table_id)

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)


def test_read_csv_gcs_bigquery_engine(
    bq_cmek, session_with_bq_cmek, scalars_df_index, gcs_folder
):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Session should have cmek set in the default query job config
    assert (
        session_with_bq_cmek.bqclient.default_query_job_config.destination_encryption_configuration.kms_key_name
        == bq_cmek
    )

    # Create a csv in gcs
    path = gcs_folder + "test_read_csv_gcs_bigquery_engine*.csv"
    scalars_df_index.to_csv(path)

    # Read the BQ table
    df = session_with_bq_cmek.read_csv(path, engine="bigquery")

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)


@pytest.mark.skip(
    reason="Internal issue 327544164, cmek does not propagate to the dataframe."
)
def test_read_pandas(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Session should have cmek set in the default query job config
    assert (
        session_with_bq_cmek.bqclient.default_query_job_config.destination_encryption_configuration.kms_key_name
        == bq_cmek
    )

    # Read the BQ table
    df = session_with_bq_cmek.read_pandas(pandas.DataFrame([[1]]))

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)
