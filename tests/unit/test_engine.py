import pytest


@pytest.mark.parametrize(
    ("bad_table_id",),
    [
        ("",),
        ("table",),
        ("dataset.table",),
    ],
)
def test_read_gbq_missing_parts(engine, bad_table_id):
    with pytest.raises(ValueError):
        engine.read_gbq(bad_table_id)
