import google.api_core.exceptions
import pytest


@pytest.mark.parametrize("missing_parts_table_id", [(""), ("table")])
def test_read_gbq_missing_parts(session, missing_parts_table_id):
    with pytest.raises(ValueError):
        session.read_gbq(missing_parts_table_id)


@pytest.mark.parametrize(
    "not_found_table_id",
    [("unknown.dataset.table"), ("project.unknown.table"), ("project.dataset.unknown")],
)
def test_read_gdb_not_found_tables(session, not_found_table_id):
    with pytest.raises(google.api_core.exceptions.NotFound):
        session.read_gbq(not_found_table_id)


@pytest.mark.parametrize(
    "good_table_id,expected", [("project.dataset.table", 1), ("dataset.table", 0)]
)
def test_read_gbq_good_tables(session, good_table_id, expected):
    df = session.read_gbq(good_table_id)
    assert len(df._block.expr._columns) == expected
