def test_read_gbq(engine, scalars_table_id, scalars_schema):
    df = engine.read_gbq(scalars_table_id)
    # TODO(swast): Test against public properties like columns or dtypes. Also,
    # check the names and data types match up.
    assert len(df._table.schema()) == len(scalars_schema)
