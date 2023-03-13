import base64
import datetime
import decimal
import pathlib
import typing
from typing import Collection

from google.cloud import bigquery
import pandas as pd
import pytest
import test_utils.prefixer

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


def load_scalars(
    dataset_id: str,
    table_id: str,
    bigquery_client: bigquery.Client,
    scalars_schema: Collection[bigquery.SchemaField],
) -> bigquery.LoadJob:
    """Create a temporary table with test data."""
    job_config = bigquery.LoadJobConfig()
    job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    job_config.schema = scalars_schema
    table_id = f"{dataset_id}.{table_id}"
    with open(DATA_DIR / "scalars.jsonl", "rb") as input_file:
        job = bigquery_client.load_table_from_file(
            input_file, table_id, job_config=job_config
        )
    # No cleanup necessary, as the surrounding dataset will delete contents.
    return typing.cast(bigquery.LoadJob, job.result())


@pytest.fixture(scope="session")
def scalars_table_id(
    dataset_id: str,
    bigquery_client: bigquery.Client,
    scalars_schema: Collection[bigquery.SchemaField],
) -> str:
    scalars_load_job = load_scalars(
        dataset_id, "scalars", bigquery_client, scalars_schema
    )
    table_ref = scalars_load_job.destination
    return f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"


@pytest.fixture(scope="session")
def scalars_table_id_2(
    dataset_id: str,
    bigquery_client: bigquery.Client,
    scalars_schema: Collection[bigquery.SchemaField],
) -> str:
    scalars_load_job = load_scalars(
        dataset_id, "scalars_too", bigquery_client, scalars_schema
    )
    table_ref = scalars_load_job.destination
    return f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"


@pytest.fixture(scope="session")
def scalars_df_no_index(
    scalars_table_id: str, session: bigframes.Session
) -> bigframes.DataFrame:
    """DataFrame pointing at test data."""
    return session.read_gbq(scalars_table_id)


@pytest.fixture(scope="session")
def scalars_df_index(scalars_df_no_index: bigframes.DataFrame) -> bigframes.DataFrame:
    """DataFrame pointing at test data."""
    return scalars_df_no_index.set_index("rowindex")


@pytest.fixture(scope="session")
def scalars_df_2_no_index(
    scalars_table_id_2: str, session: bigframes.Session
) -> bigframes.DataFrame:
    """DataFrame pointing at test data."""
    return session.read_gbq(scalars_table_id_2)


@pytest.fixture(scope="session")
def scalars_df_2_index(
    scalars_df_2_no_index: bigframes.DataFrame,
) -> bigframes.DataFrame:
    """DataFrame pointing at test data."""
    return scalars_df_2_no_index.set_index("rowindex")


@pytest.fixture(scope="session")
def scalars_pandas_df_default_index() -> pd.DataFrame:
    """pandas.DataFrame pointing at test data."""

    df = pd.read_json(
        DATA_DIR / "scalars.jsonl",
        lines=True,
        # Convert default pandas dtypes to match BigFrames dtypes.
        dtype={
            "bool_col": pd.BooleanDtype(),
            "int64_col": pd.Int64Dtype(),
            "int64_too": pd.Int64Dtype(),
            "float64_col": pd.Float64Dtype(),
            "rowindex": pd.Int64Dtype(),
        },
    )
    df["bytes_col"] = df["bytes_col"].apply(
        lambda value: base64.b64decode(value) if value else value
    )
    # TODO(swast): Use db_dtypes.DateDtype() for BigQuery DATE columns. Needs
    # microsecond precision support:
    # https://github.com/googleapis/python-db-dtypes-pandas/issues/47
    df["date_col"] = pd.to_datetime(df["date_col"])
    df["datetime_col"] = pd.to_datetime(df["datetime_col"])
    # TODO(swast): Use db_dtypes.TimeDtype() for BigQuery TIME columns.
    df["time_col"] = df["time_col"].apply(
        lambda value: datetime.time.fromisoformat(value) if value else value
    )
    # TODO(swast): Ensure BigQuery TIMESTAMP columns have UTC timezone.
    df["timestamp_col"] = pd.to_datetime(df["timestamp_col"])
    df["numeric_col"] = df["numeric_col"].apply(
        lambda value: decimal.Decimal(str(value)) if value else None  # type: ignore
    )
    return df


@pytest.fixture(scope="session")
def scalars_pandas_df_index(
    scalars_pandas_df_default_index: pd.DataFrame,
) -> pd.DataFrame:
    """pandas.DataFrame pointing at test data."""
    return scalars_pandas_df_default_index.set_index("rowindex").sort_index()


@pytest.fixture(scope="session", params=("index", "no_index"))
def scalars_dfs(
    request,
    scalars_df_no_index,
    scalars_df_index,
    scalars_pandas_df_default_index,
    scalars_pandas_df_index,
):
    if request.param == "index":
        return scalars_df_index, scalars_pandas_df_index
    else:
        return scalars_df_no_index, scalars_pandas_df_default_index
