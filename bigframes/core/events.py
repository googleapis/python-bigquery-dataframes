# Copyright 2025 Google LLC
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

from __future__ import annotations

import dataclasses
import datetime
import threading
from typing import List, Optional
import weakref

import google.cloud.bigquery._job_helpers
import google.cloud.bigquery.job.query
import google.cloud.bigquery.table


@dataclasses.dataclass(frozen=True)
class Subscriber:
    callback_ref: weakref.ref
    # TODO(tswast): Add block_id to allow filter in context managers.


class Publisher:
    def __init__(self):
        self._subscribers: List[Subscriber] = []
        self._subscribers_lock = threading.Lock()

    def subscribe(self, callback):
        subscriber = Subscriber(callback_ref=weakref.ref(callback))

        with self._subscribers_lock:
            # TODO(tswast): Add block_id to allow filter in context managers.
            self._subscribers.append(subscriber)

    def send(self, event: Event):
        to_delete = []
        to_call = []

        with self._subscribers_lock:
            for sid, subscriber in enumerate(self._subscribers):
                callback = subscriber.callback_ref()

                if callback is None:
                    to_delete.append(sid)
                else:
                    # TODO(tswast): Add if statement for block_id to allow filter
                    # in context managers.
                    to_call.append(callback)

            for sid in reversed(to_delete):
                del self._subscribers[sid]

        for callback in to_call:
            callback(event)


publisher = Publisher()


class Event:
    pass


class ExecutionStarted(Event):
    pass


class ExecutionRunning(Event):
    pass


class ExecutionStopped(Event):
    pass


@dataclasses.dataclass(frozen=True)
class BigQuerySentEvent(ExecutionStarted):
    """Query sent to BigQuery."""

    query: str
    billing_project: Optional[str] = None
    location: Optional[str] = None
    job_id: Optional[str] = None
    request_id: Optional[str] = None

    @classmethod
    def from_bqclient(cls, event: google.cloud.bigquery._job_helpers.QuerySentEvent):
        return cls(
            query=event.query,
            billing_project=event.billing_project,
            location=event.location,
            job_id=event.job_id,
            request_id=event.request_id,
        )


@dataclasses.dataclass(frozen=True)
class BigQueryRetryEvent(ExecutionRunning):
    """Query sent another time because the previous attempt failed."""

    query: str
    billing_project: Optional[str] = None
    location: Optional[str] = None
    job_id: Optional[str] = None
    request_id: Optional[str] = None

    @classmethod
    def from_bqclient(cls, event: google.cloud.bigquery._job_helpers.QueryRetryEvent):
        return cls(
            query=event.query,
            billing_project=event.billing_project,
            location=event.location,
            job_id=event.job_id,
            request_id=event.request_id,
        )


@dataclasses.dataclass(frozen=True)
class BigQueryReceivedEvent(ExecutionRunning):
    """Query received and acknowledged by the BigQuery API."""

    billing_project: Optional[str] = None
    location: Optional[str] = None
    job_id: Optional[str] = None
    statement_type: Optional[str] = None
    state: Optional[str] = None
    query_plan: Optional[list[google.cloud.bigquery.job.query.QueryPlanEntry]] = None
    created: Optional[datetime.datetime] = None
    started: Optional[datetime.datetime] = None
    ended: Optional[datetime.datetime] = None

    @classmethod
    def from_bqclient(
        cls, event: google.cloud.bigquery._job_helpers.QueryReceivedEvent
    ):
        return cls(
            billing_project=event.billing_project,
            location=event.location,
            job_id=event.job_id,
            statement_type=event.statement_type,
            state=event.state,
            query_plan=event.query_plan,
            created=event.created,
            started=event.started,
            ended=event.ended,
        )


@dataclasses.dataclass(frozen=True)
class BigQueryFinishedEvent(ExecutionStopped):
    """Query finished successfully."""

    billing_project: Optional[str] = None
    location: Optional[str] = None
    query_id: Optional[str] = None
    job_id: Optional[str] = None
    destination: Optional[google.cloud.bigquery.table.TableReference] = None
    total_rows: Optional[int] = None
    total_bytes_processed: Optional[int] = None
    slot_millis: Optional[int] = None
    created: Optional[datetime.datetime] = None
    started: Optional[datetime.datetime] = None
    ended: Optional[datetime.datetime] = None

    @classmethod
    def from_bqclient(
        cls, event: google.cloud.bigquery._job_helpers.QueryFinishedEvent
    ):
        return cls(
            billing_project=event.billing_project,
            location=event.location,
            query_id=event.query_id,
            job_id=event.job_id,
            destination=event.destination,
            total_rows=event.total_rows,
            total_bytes_processed=event.total_bytes_processed,
            slot_millis=event.slot_millis,
            created=event.created,
            started=event.started,
            ended=event.ended,
        )


@dataclasses.dataclass(frozen=True)
class BigQueryUnknownEvent(ExecutionRunning):
    """Got unknown event from the BigQuery client library."""

    # TODO: should we just skip sending unknown events?

    event: object

    @classmethod
    def from_bqclient(cls, event):
        return cls(event)
