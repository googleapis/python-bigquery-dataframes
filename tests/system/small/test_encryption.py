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
import pandas
import pytest

import bigframes


@pytest.fixture(scope="module")
def bq_cmek() -> str:
    """Customer managed encryption key to encrypt BigQuery data at rest.

    This is of the form projects/PROJECT_ID/locations/LOCATION/keyRings/KEY_RING/cryptoKeys/KEY

    See https://cloud.google.com/bigquery/docs/customer-managed-encryption for steps.
    """

    # NOTE: Set a key here for local testing.
    # We are keeping this empty by default, in which case the dependent tests
    # are skipped.
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


def test_session_default_configs(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Session should have cmek set in the default query and load job configs
    assert (
        session_with_bq_cmek.bqclient.default_query_job_config.destination_encryption_configuration.kms_key_name
        == bq_cmek
    )
    assert (
        session_with_bq_cmek.bqclient.default_load_job_config.destination_encryption_configuration.kms_key_name
        == bq_cmek
    )


def test_session_query_job(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    query_job = session_with_bq_cmek.bqclient.query("SELECT 123")
    query_job.result()

    assert query_job.destination_encryption_configuration.kms_key_name.startswith(
        bq_cmek
    )

    # The result table should exist with the intended encryption
    table = session_with_bq_cmek.bqclient.get_table(query_job.destination)
    assert table.encryption_configuration.kms_key_name == bq_cmek


def test_session_load_job(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Session should have cmek set in the default query and load job configs
    load_table = bigframes.session._io.bigquery.random_table(
        session_with_bq_cmek._anonymous_dataset
    )
    load_job = session_with_bq_cmek.bqclient.load_table_from_dataframe(
        pandas.DataFrame({"col0": [1, 2, 3]}),
        load_table,
        job_config=bigquery.LoadJobConfig(
            schema=[bigquery.SchemaField("col0", bigquery.enums.SqlTypeNames.INT64)]
        ),
    )
    load_job.result()

    assert load_job.destination == load_table
    assert load_job.destination_encryption_configuration.kms_key_name.startswith(
        bq_cmek
    )

    # The result table should exist with the intended encryption
    table = session_with_bq_cmek.bqclient.get_table(load_job.destination)
    assert table.encryption_configuration.kms_key_name == bq_cmek


def test_read_gbq(bq_cmek, session_with_bq_cmek, scalars_table_id):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Read the BQ table
    df = session_with_bq_cmek.read_gbq(scalars_table_id)

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)


def test_df_apis(bq_cmek, session_with_bq_cmek, scalars_table_id):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Read a BQ table and assert encryption
    df = session_with_bq_cmek.read_gbq(scalars_table_id)

    # Perform a few dataframe operations and assert assertion
    df1 = df.dropna()
    _assert_bq_table_is_encrypted(df1, bq_cmek, session_with_bq_cmek)

    df2 = df1.head()
    _assert_bq_table_is_encrypted(df2, bq_cmek, session_with_bq_cmek)


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("bigquery", id="bq_engine"),
        pytest.param(
            None,
            id="default_engine",
            marks=pytest.mark.skip(
                reason="Internal issue 327544164, cmek does not propagate to the dataframe."
            ),
        ),
    ],
)
def test_read_csv_gcs(
    bq_cmek, session_with_bq_cmek, scalars_df_index, gcs_folder, engine
):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Create a csv in gcs
    write_path = gcs_folder + "test_read_csv_gcs_bigquery_engine*.csv"
    read_path = (
        write_path.replace("*", "000000000000") if engine is None else write_path
    )
    scalars_df_index.to_csv(write_path)

    # Read the BQ table
    df = session_with_bq_cmek.read_csv(read_path, engine=engine)

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)


def test_to_gbq(bq_cmek, session_with_bq_cmek, scalars_table_id):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Read a BQ table and assert encryption
    df = session_with_bq_cmek.read_gbq(scalars_table_id)
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)

    # Modify the dataframe and assert assertion
    df = df.dropna().head()
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)

    # Write the result to BQ and assert assertion
    output_table_id = df.to_gbq()
    output_table = session_with_bq_cmek.bqclient.get_table(output_table_id)
    assert output_table.encryption_configuration.kms_key_name == bq_cmek


@pytest.mark.skip(
    reason="Internal issue 327544164, cmek does not propagate to the dataframe."
)
def test_read_pandas(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Read a pandas dataframe
    df = session_with_bq_cmek.read_pandas(pandas.DataFrame([1]))

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)


def test_read_pandas_large(bq_cmek, session_with_bq_cmek):
    if not bq_cmek:
        pytest.skip("no cmek set for testing")

    # Read a pandas dataframe large enough to trigger a BQ load job
    df = session_with_bq_cmek.read_pandas(pandas.DataFrame(range(10_000)))

    # Assert encryption
    _assert_bq_table_is_encrypted(df, bq_cmek, session_with_bq_cmek)
