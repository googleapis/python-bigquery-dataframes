# Copyright 2026 Google LLC
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
import unittest.mock

import google.cloud.bigquery as bigquery
import pandas

import bigframes.pandas as bpd
import bigframes.session
import bigframes.session.metrics as metrics

NOW = datetime.datetime.now(datetime.timezone.utc)


def test_pandas_job_history(monkeypatch):
    query_job = unittest.mock.create_autospec(bigquery.QueryJob, instance=True)
    query_job.job_id = "job_pandas"
    query_job.location = "US"
    query_job.created = NOW
    query_job.started = NOW
    query_job.ended = NOW + datetime.timedelta(seconds=1)
    query_job.state = "DONE"
    query_job.total_bytes_processed = 1024
    query_job.slot_millis = 100
    query_job.job_type = "query"
    query_job.error_result = None
    query_job.cache_hit = False
    query_job.query = "SELECT 1"
    query_job.configuration.dry_run = False

    # Mock the clients provider to avoid actual BQ client creation
    clients_provider = unittest.mock.create_autospec(
        bigframes.session.clients.ClientsProvider
    )
    bq_client = unittest.mock.create_autospec(bigquery.Client)
    bq_client.project = "test-project"
    bq_client.default_query_job_config = bigquery.QueryJobConfig()
    bq_client.query_and_wait.return_value = iter([[NOW]])
    clients_provider.bqclient = bq_client

    session = bigframes.session.Session(clients_provider=clients_provider)

    # Mock get_global_session to return our session
    monkeypatch.setattr(
        bigframes.core.global_session, "get_global_session", lambda: session
    )

    session._metrics.count_job_stats(query_job=query_job)

    df = bpd.job_history()

    assert isinstance(df, pandas.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["job_id"] == "job_pandas"


def test_session_job_history():
    query_job = unittest.mock.create_autospec(bigquery.QueryJob, instance=True)
    query_job.job_id = "job1"
    query_job.location = "US"
    query_job.created = NOW
    query_job.started = NOW
    query_job.ended = NOW + datetime.timedelta(seconds=1)
    query_job.state = "DONE"
    query_job.total_bytes_processed = 1024
    query_job.slot_millis = 100
    query_job.job_type = "query"
    query_job.error_result = None
    query_job.cache_hit = False
    query_job.query = "SELECT 1"
    query_job.configuration.dry_run = False

    # Mock the clients provider to avoid actual BQ client creation
    clients_provider = unittest.mock.create_autospec(
        bigframes.session.clients.ClientsProvider
    )
    # We need to mock bqclient specifically as it's accessed during Session init
    bq_client = unittest.mock.create_autospec(bigquery.Client)
    bq_client.project = "test-project"
    bq_client.default_query_job_config = bigquery.QueryJobConfig()
    # Mock clock sync query
    bq_client.query_and_wait.return_value = iter([[NOW]])
    clients_provider.bqclient = bq_client

    session = bigframes.session.Session(clients_provider=clients_provider)
    session._metrics.count_job_stats(query_job=query_job)

    df = session.job_history()

    assert isinstance(df, pandas.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["job_id"] == "job1"
    assert "query_id" in df.columns
    assert "creation_time" in df.columns


def test_job_history_with_query_job():
    query_job = unittest.mock.create_autospec(bigquery.QueryJob, instance=True)
    query_job.job_id = "job1"
    query_job.location = "US"
    query_job.created = NOW
    query_job.started = NOW + datetime.timedelta(seconds=1)
    query_job.ended = NOW + datetime.timedelta(seconds=3)
    query_job.state = "DONE"
    query_job.total_bytes_processed = 1024
    query_job.slot_millis = 100
    query_job.job_type = "query"
    query_job.error_result = None
    query_job.cache_hit = False
    query_job.query = "SELECT 1"
    query_job.configuration.dry_run = False

    execution_metrics = metrics.ExecutionMetrics()
    execution_metrics.count_job_stats(query_job=query_job)

    assert len(execution_metrics.jobs) == 1
    job = execution_metrics.jobs[0]
    assert job.job_id == "job1"
    assert job.status == "DONE"
    assert job.total_bytes_processed == 1024
    assert job.duration_seconds == 3.0


def test_job_history_with_row_iterator():
    row_iterator = unittest.mock.create_autospec(
        bigquery.table.RowIterator, instance=True
    )
    row_iterator.job_id = "job2"
    row_iterator.query_id = "query2"
    row_iterator.location = "US"
    row_iterator.created = NOW
    row_iterator.started = NOW + datetime.timedelta(seconds=1)
    row_iterator.ended = NOW + datetime.timedelta(seconds=2)
    row_iterator.total_bytes_processed = 512
    row_iterator.slot_millis = 50
    row_iterator.cache_hit = True
    row_iterator.query = "SELECT 2"

    execution_metrics = metrics.ExecutionMetrics()
    execution_metrics.count_job_stats(row_iterator=row_iterator)

    assert len(execution_metrics.jobs) == 1
    job = execution_metrics.jobs[0]
    assert job.job_id == "job2"
    assert job.query_id == "query2"
    assert job.status == "DONE"
    assert job.cached is True
    assert job.duration_seconds == 2.0


def test_job_history_with_load_job():
    load_job = unittest.mock.create_autospec(bigquery.LoadJob, instance=True)
    load_job.job_id = "job3"
    load_job.location = "US"
    load_job.created = NOW
    load_job.started = NOW
    load_job.ended = NOW + datetime.timedelta(seconds=5)
    load_job.state = "DONE"
    load_job.job_type = "load"
    load_job.error_result = None
    load_job.configuration.dry_run = False
    load_job.output_rows = 100
    load_job.input_files = 1
    load_job.input_bytes = 1024
    load_job.destination = bigquery.TableReference.from_string("project.dataset.table")
    load_job.source_uris = ["gs://bucket/file.csv"]
    load_job.configuration.source_format = "CSV"

    execution_metrics = metrics.ExecutionMetrics()
    execution_metrics.count_job_stats(query_job=load_job)

    assert len(execution_metrics.jobs) == 1
    job = execution_metrics.jobs[0]
    assert job.job_id == "job3"
    assert job.job_type == "load"
    assert job.duration_seconds == 5.0
    assert job.output_rows == 100
    assert job.input_files == 1
    assert job.input_bytes == 1024
    assert job.destination_table == "project.dataset.table"
    assert job.source_uris == ["gs://bucket/file.csv"]
    assert job.source_format == "CSV"
