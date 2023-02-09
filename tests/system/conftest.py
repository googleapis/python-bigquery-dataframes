import pathlib
import typing
from typing import Collection

import pandas as pd
import pytest
import test_utils.prefixer
from google.cloud import bigquery

import bigframes

CURRENT_DIR = pathlib.Path(__file__).parent
DATA_DIR = CURRENT_DIR.parent / "data"
prefixer = test_utils.prefixer.Prefixer("bigframes", "tests/system")


@pytest.fixture(scope="session")
def bigquery_client(session: bigframes.Session) -> bigquery.Client:
    return session.bqclient


@pytest.fixture(scope="session")
def session() -> bigframes.Session:
    return bigframes.Session()


@pytest.fixture(scope="session", autouse=True)
def cleanup_datasets(bigquery_client: bigquery.Client) -> None:
    """Cleanup any datasets that were created but not cleaned up."""
    for dataset in bigquery_client.list_datasets():
        if prefixer.should_cleanup(dataset.dataset_id):
            bigquery_client.delete_dataset(
                dataset, delete_contents=True, not_found_ok=True
            )


@pytest.fixture(scope="session")
def dataset_id(bigquery_client: bigquery.Client):
    """Create (and cleanup) a temporary dataset."""
    project_id = bigquery_client.project
    dataset_id = f"{project_id}.{prefixer.create_prefix()}_dataset_id"
    dataset = bigquery.Dataset(dataset_id)
    bigquery_client.create_dataset(dataset)
    yield dataset_id
    bigquery_client.delete_dataset(dataset, delete_contents=True)


@pytest.fixture(scope="session")
def scalars_schema(bigquery_client: bigquery.Client):
    # TODO(swast): Add missing scalar data types such as BIGNUMERIC.
    # See also: https://github.com/ibis-project/ibis-bigquery/pull/67
    schema = bigquery_client.schema_from_json(DATA_DIR / "scalars_schema.json")
    return tuple(schema)


@pytest.fixture(scope="session")
def scalars_load_job(
    dataset_id: str,
    bigquery_client: bigquery.Client,
    scalars_schema: Collection[bigquery.SchemaField],
) -> bigquery.LoadJob:
    """Create a temporary table with test data."""
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.schema = scalars_schema
    table_id = f"{dataset_id}.scalars"
    with open(DATA_DIR / "scalars.jsonl", "rb") as input_file:
        job = bigquery_client.load_table_from_file(
            input_file, table_id, job_config=job_config
        )
    # No cleanup necessary, as the surrounding dataset will delete contents.
    return typing.cast(bigquery.LoadJob, job.result())


@pytest.fixture(scope="session")
def scalars_table_id(scalars_load_job: bigquery.LoadJob) -> str:
    table_ref = scalars_load_job.destination
    return f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"


@pytest.fixture(scope="session")
def scalars_df(
    scalars_table_id: str, session: bigframes.Session
) -> bigframes.DataFrame:
    """DataFrame pointing at test data."""
    return session.read_gbq(scalars_table_id)


@pytest.fixture(scope="session")
def scalars_pandas_df() -> pd.DataFrame:
    """pandas.DataFrame pointing at test data."""
    return pd.read_json(DATA_DIR / "scalars.jsonl", lines=True)
