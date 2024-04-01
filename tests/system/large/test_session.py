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

import pytest

import bigframes
import bigframes.pandas as bpd


@pytest.mark.parametrize(
    ("query_or_table", "index_col"),
    [
        pytest.param(
            "bigquery-public-data.patents_view.ipcr_201708",
            (),
            id="1g_table_w_default_index",
        ),
        pytest.param(
            "bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2011",
            (),
            id="30g_table_w_default_index",
        ),
        # TODO(chelsealin): Disable the long run tests until we have propertily
        # ordering support to avoid materializating any data.
        # # Adding default index to large tables would take much longer time,
        # # e.g. ~5 mins for a 100G table, ~20 mins for a 1T table.
        # pytest.param(
        #     "bigquery-public-data.stackoverflow.post_history",
        #     ["id"],
        #     id="100g_table_w_unique_column_index",
        # ),
        # pytest.param(
        #     "bigquery-public-data.wise_all_sky_data_release.all_wise",
        #     ["cntr"],
        #     id="1t_table_w_unique_column_index",
        # ),
    ],
)
def test_read_gbq_for_large_tables(
    session: bigframes.Session, query_or_table, index_col
):
    """Verify read_gbq() is able to read large tables."""
    df = session.read_gbq(query_or_table, index_col=index_col)
    assert len(df.columns) != 0


# this test is prohibitively slow on our test project
@pytest.mark.skip
def test_close(session):
    session_id = session.session_id

    # we will create two tables and confirm that they are deleted
    # when the session is closed

    bqclient = session.bqclient
    dataset = session._anonymous_dataset
    expiration = (
        datetime.datetime.now(datetime.timezone.utc)
        + bigframes.constants.DEFAULT_EXPIRATION
    )
    bigframes.session._io.bigquery.create_temp_table(
        bqclient, session_id, dataset, expiration
    )
    bigframes.session._io.bigquery.create_temp_table(
        bqclient, session_id, dataset, expiration
    )
    tables_before = bqclient.list_tables(dataset, page_size=1000)
    tables_before_count = len(list(tables_before))
    assert tables_before_count >= 2

    session.close()

    tables_after = bqclient.list_tables(dataset, page_size=1000)
    assert len(list(tables_after)) <= tables_before_count - 2


# this test is prohibitively slow on our test project
@pytest.mark.skip
def test_pandas_close_session():
    session = bpd.get_global_session()
    session_id = session.session_id

    # we will create two tables and confirm that they are deleted
    # when the session is closed

    bqclient = session.bqclient
    dataset = session._anonymous_dataset
    expiration = (
        datetime.datetime.now(datetime.timezone.utc)
        + bigframes.constants.DEFAULT_EXPIRATION
    )
    bigframes.session._io.bigquery.create_temp_table(
        bqclient, session_id, dataset, expiration
    )
    bigframes.session._io.bigquery.create_temp_table(
        bqclient, session_id, dataset, expiration
    )
    tables_before = bqclient.list_tables(dataset, page_size=1000)
    tables_before_count = len(list(tables_before))
    assert tables_before_count >= 2

    bpd.close_session(session_id=session_id)

    tables_after = bqclient.list_tables(dataset, page_size=1000)
    assert len(list(tables_after)) <= tables_before_count - 2
