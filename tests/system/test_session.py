import google.api_core.exceptions
import pytest

import bigframes


def test_read_gbq(session: bigframes.Session, scalars_table_id, scalars_schema):
    df = session.read_gbq(scalars_table_id)
    # TODO(swast): Test against public properties like columns or dtypes. Also,
    # check the names and data types match up.
    assert len(df._block.expr._columns) == len(scalars_schema)


def test_read_gbq_w_col_order(session, scalars_table_id, scalars_schema):
    columns = list(column.name for column in scalars_schema)
    df = session.read_gbq(scalars_table_id, col_order=columns)
    assert len(df._block.expr._columns) == len(scalars_schema)

    df = session.read_gbq(scalars_table_id, col_order=[columns[0]])
    assert len(df._block.expr._columns) == 1

    with pytest.raises(ValueError):
        df = session.read_gbq(scalars_table_id, col_order=["unknown"])


def test_session_id(session):
    assert session._session_id is not None

    # BQ client always runs query within the opened session.
    query_job = session.bqclient.query("SELECT 1")
    assert query_job.session_info.session_id == session._session_id


def test_to_close_session(session):
    assert session._session_id is not None
    session.close()
    assert session._session_id is None

    # Session has expired and is no longer available.
    with pytest.raises(google.api_core.exceptions.BadRequest):
        query_job = session.bqclient.query("SELECT 1")
        query_job.result()  # blocks until finished
