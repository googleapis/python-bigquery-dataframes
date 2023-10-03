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

import datetime
from typing import Iterable
from unittest import mock

import google.cloud.bigquery as bigquery
import pytest

import bigframes.core.io


def test_create_job_configs_is_none():
    mock_job_config = mock.create_autospec(None)
    api_methods = ["df-agg", "series-mode"]
    labels, _ = bigframes.core.io.create_job_configs_labels(
        job_config=mock_job_config, api_methods=api_methods
    )
    expected_dict = {"bigframes-api-0": "df-agg", "bigframes-api-1": "series-mode"}
    assert labels is not None
    assert labels == expected_dict


def test_create_job_configs_labels_is_none():
    mock_job_config = mock.create_autospec(bigquery.QueryJobConfig())
    api_methods = ["df-agg", "series-mode"]
    labels, _ = bigframes.core.io.create_job_configs_labels(
        job_config=mock_job_config, api_methods=api_methods
    )
    expected_dict = {"bigframes-api-0": "df-agg", "bigframes-api-1": "series-mode"}
    assert labels is not None
    assert labels == expected_dict


def test_create_job_configs_labels_length_limit_not_met():
    mock_job_config = mock.create_autospec(bigquery.QueryJobConfig())
    mock_job_config.labels = {
        "bigframes-api-0": "test0",
        "bigframes-api-1": "test1",
        "bigframes-api-2": "test2",
    }
    api_methods = ["df-agg", "series-mode"]
    labels, _ = bigframes.core.io.create_job_configs_labels(
        job_config=mock_job_config, api_methods=api_methods
    )
    expected_dict = {
        "bigframes-api-0": "test0",
        "bigframes-api-1": "test1",
        "bigframes-api-2": "test2",
        "bigframes-api-3": "df-agg",
        "bigframes-api-4": "series-mode",
    }
    assert labels is not None
    assert len(labels) == 5
    assert labels == expected_dict


def test_create_job_configs_labels_length_limit_met():
    mock_job_config = mock.create_autospec(bigquery.QueryJobConfig())
    cur_labels = {}
    for i in range(63):
        key = f"bigframes-api{i}"
        value = f"test{i}"
        cur_labels[key] = value
    # If cur_labels length is 63, we can only add one label from api_methods
    mock_job_config.labels = cur_labels
    api_methods = ["df-agg", "series-mode"]
    labels, _ = bigframes.core.io.create_job_configs_labels(
        job_config=mock_job_config, api_methods=api_methods
    )
    assert labels is not None
    assert len(labels) == 64
    assert "df-agg" not in labels.values()
    assert "series-mode" in labels.values()


def test_create_snapshot_sql_doesnt_timetravel_anonymous_datasets():
    table_ref = bigquery.TableReference.from_string(
        "my-test-project._e8166e0cdb.anonbb92cd"
    )

    sql = bigframes.core.io.create_snapshot_sql(
        table_ref, datetime.datetime.now(datetime.timezone.utc)
    )

    # Anonymous query results tables don't support time travel.
    assert "SYSTEM_TIME" not in sql

    # Need fully-qualified table name.
    assert "`my-test-project`.`_e8166e0cdb`.`anonbb92cd`" in sql


def test_create_snapshot_sql_doesnt_timetravel_session_datasets():
    table_ref = bigquery.TableReference.from_string("my-test-project._session.abcdefg")

    sql = bigframes.core.io.create_snapshot_sql(
        table_ref, datetime.datetime.now(datetime.timezone.utc)
    )

    # We aren't modifying _SESSION tables, so don't use time travel.
    assert "SYSTEM_TIME" not in sql

    # Don't need the project ID for _SESSION tables.
    assert "my-test-project" not in sql


@pytest.mark.parametrize(
    ("schema", "expected"),
    (
        (
            [bigquery.SchemaField("My Column", "INTEGER")],
            "`My Column` INT64",
        ),
        (
            [
                bigquery.SchemaField("My Column", "INTEGER"),
                bigquery.SchemaField("Float Column", "FLOAT"),
                bigquery.SchemaField("Bool Column", "BOOLEAN"),
            ],
            "`My Column` INT64, `Float Column` FLOAT64, `Bool Column` BOOL",
        ),
        (
            [
                bigquery.SchemaField("My Column", "INTEGER", mode="REPEATED"),
                bigquery.SchemaField("Float Column", "FLOAT", mode="REPEATED"),
                bigquery.SchemaField("Bool Column", "BOOLEAN", mode="REPEATED"),
            ],
            "`My Column` ARRAY<INT64>, `Float Column` ARRAY<FLOAT64>, `Bool Column` ARRAY<BOOL>",
        ),
        (
            [
                bigquery.SchemaField(
                    "My Column",
                    "RECORD",
                    mode="REPEATED",
                    fields=(
                        bigquery.SchemaField("Float Column", "FLOAT", mode="REPEATED"),
                        bigquery.SchemaField("Bool Column", "BOOLEAN", mode="REPEATED"),
                        bigquery.SchemaField(
                            "Nested Column",
                            "RECORD",
                            fields=(bigquery.SchemaField("Int Column", "INTEGER"),),
                        ),
                    ),
                ),
            ],
            (
                "`My Column` ARRAY<STRUCT<"
                + "`Float Column` ARRAY<FLOAT64>,"
                + " `Bool Column` ARRAY<BOOL>,"
                + " `Nested Column` STRUCT<`Int Column` INT64>>>"
            ),
        ),
    ),
)
def test_bq_schema_to_sql(schema: Iterable[bigquery.SchemaField], expected: str):
    pass
