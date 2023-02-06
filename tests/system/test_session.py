def test_read_gbq(session, scalars_table_id, scalars_schema):
    df = session.read_gbq(scalars_table_id)
    # TODO(swast): Test against public properties like columns or dtypes. Also,
    # check the names and data types match up.
    assert len(df._expr._columns) == len(scalars_schema)
